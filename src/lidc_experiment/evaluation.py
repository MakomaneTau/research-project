"""Evaluation of frozen nodule predictions with patient-level uncertainty.

This module never fits models or selects their hyperparameters.  The positive
class is high radiologist-rated suspicion, not pathology-confirmed malignancy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

METRICS = (
    "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
    "sensitivity", "specificity", "precision", "f1", "brier_score",
)
REQUIRED_COLUMNS = {"patient_id", "nodule_id", "label", "model", "score", "probability"}


def choose_threshold(labels, scores, target_sensitivity: float = 0.90) -> float:
    """Highest observed score reaching the declared validation sensitivity.

    Decisions use >=, including every observation tied at the cutoff.
    """
    y = np.asarray(labels)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not len(y):
        raise ValueError("Labels and scores must be nonempty, matching vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Threshold selection requires binary labels and finite scores")
    if not 0 < target_sensitivity <= 1:
        raise ValueError("Target sensitivity must be in (0, 1]")
    positive = y == 1
    if not positive.any():
        raise ValueError("A sensitivity threshold requires positive validation cases")
    for threshold in np.unique(score)[::-1]:
        if np.mean(score[positive] >= threshold) >= target_sensitivity:
            return float(threshold)
    raise RuntimeError("No attainable sensitivity threshold")


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Require complete, paired, unambiguous predictions from every model."""
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty or predictions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Prediction records must be nonempty and complete")
    data = predictions.copy()
    for column in ("patient_id", "nodule_id", "model"):
        data[column] = data[column].astype(str)
    if not data["label"].isin([0, 1]).all():
        raise ValueError("Only low=0 and high=1 suspicion labels are permitted")
    if data.duplicated(["model", "nodule_id"]).any():
        raise ValueError("Duplicate nodule predictions within a model")
    values = data[["score", "probability"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All model scores and probabilities must be finite")
    if ((values[:, 1] < 0) | (values[:, 1] > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    reference = None
    for name, group in data.groupby("model", sort=True):
        keys = group.sort_values("nodule_id")[["nodule_id", "patient_id", "label"]].reset_index(drop=True)
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise ValueError(f"Incomplete or inconsistent common prediction cohort for {name}")
    return data.sort_values(["model", "nodule_id"]).reset_index(drop=True)


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def classification_metrics(labels, scores, probabilities, threshold: float, sample_weight=None) -> dict:
    """Calculate metrics, retaining undefined denominators as NaN."""
    y = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    weight = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    decision = scores >= threshold
    tp = float(weight[(y == 1) & decision].sum())
    fn = float(weight[(y == 1) & ~decision].sum())
    tn = float(weight[(y == 0) & ~decision].sum())
    fp = float(weight[(y == 0) & decision].sum())
    sensitivity, specificity = _divide(tp, tp + fn), _divide(tn, tn + fp)
    both_classes = len(np.unique(y[weight > 0])) == 2
    return {
        "n_nodules": int(len(y)), "positive_nodules": int((y == 1).sum()),
        "prevalence": _divide(float(weight[y == 1].sum()), float(weight.sum())),
        "threshold": float(threshold), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "roc_auc": float(roc_auc_score(y, scores, sample_weight=weight)) if both_classes else float("nan"),
        "average_precision": float(average_precision_score(y, scores, sample_weight=weight)) if np.any((y == 1) & (weight > 0)) else float("nan"),
        "accuracy": _divide(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": _divide(tp, tp + fp), "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "brier_score": float(brier_score_loss(y, probability, sample_weight=weight)),
    }


def metrics_table(predictions: pd.DataFrame, thresholds: Mapping[str, float], *,
                  patient_weighted: bool = False, probability_threshold: bool = False) -> pd.DataFrame:
    data = validate_predictions(predictions)
    rows = []
    for name, group in data.groupby("model", sort=True):
        if name not in thresholds:
            raise ValueError(f"No frozen threshold supplied for {name}")
        weights = None
        if patient_weighted:
            weights = 1.0 / group.groupby("patient_id")["nodule_id"].transform("count").to_numpy()
        score = group["probability"] if probability_threshold else group["score"]
        threshold = 0.50 if probability_threshold else float(thresholds[name])
        row = classification_metrics(group["label"], score, group["probability"], threshold, weights)
        row.update(model=name, n_patients=int(group["patient_id"].nunique()),
                   weighting="patient" if patient_weighted else "nodule",
                   threshold_basis="probability" if probability_threshold else "native_score")
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_indices(patient_ids, rng: np.random.Generator) -> np.ndarray:
    """Draw patients with replacement and explicitly retain draw multiplicities."""
    ids = np.asarray(patient_ids, dtype=str)
    patients = np.unique(ids)
    if not len(patients):
        raise ValueError("Cannot bootstrap an empty patient list")
    members = {patient: np.flatnonzero(ids == patient) for patient in patients}
    return np.concatenate([members[patient] for patient in rng.choice(patients, len(patients), replace=True)])


def paired_patient_bootstrap(predictions: pd.DataFrame, thresholds: Mapping[str, float], comparator: str,
                             replicates: int = 2000, max_attempts: int = 20000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    data = validate_predictions(predictions)
    by_model = {name: group.sort_values("nodule_id").reset_index(drop=True)
                for name, group in data.groupby("model", sort=True)}
    if "cnn" not in by_model or comparator not in by_model:
        raise ValueError("Primary comparison requires CNN and the selected traditional model")
    if replicates < 0 or max_attempts < replicates:
        raise ValueError("Bootstrap attempts must cover the requested nonnegative replicates")
    base = next(iter(by_model.values()))
    arrays = {name: (g.label.to_numpy(int), g.score.to_numpy(float), g.probability.to_numpy(float))
              for name, g in by_model.items()}
    samples = {name: {metric: [] for metric in METRICS} for name in by_model}
    differences = {metric: [] for metric in ("roc_auc", "sensitivity", "specificity")}
    rng = np.random.default_rng(seed)
    valid = attempts = rejected = 0
    while valid < replicates and attempts < max_attempts:
        attempts += 1
        indices = cluster_bootstrap_indices(base.patient_id.to_numpy(), rng)
        if len(np.unique(base.label.to_numpy()[indices])) < 2:
            rejected += 1
            continue
        results = {}
        for name, (y, score, probability) in arrays.items():
            result = classification_metrics(y[indices], score[indices], probability[indices], thresholds[name])
            results[name] = result
            for metric in METRICS:
                samples[name][metric].append(result[metric])
        for metric in differences:
            differences[metric].append(results["cnn"][metric] - results[comparator][metric])
        valid += 1
    point = metrics_table(data, thresholds).set_index("model")
    rows = []
    for name, metrics in samples.items():
        for metric, values in metrics.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            lower, upper = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            rows.append({"model": name, "metric": metric, "estimate": point.loc[name, metric],
                         "lower_95": lower, "upper_95": upper, "valid_metric_replicates": len(finite)})
    difference_rows = {}
    for metric, values in differences.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        lower, upper = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        difference_rows[metric] = {
            "estimate": float(point.loc["cnn", metric] - point.loc[comparator, metric]),
            "lower_95": float(lower), "upper_95": float(upper), "valid_metric_replicates": int(len(finite)),
        }
    auc = difference_rows["roc_auc"]
    reliable = valid == replicates and valid > 0
    interpretation = "unavailable"
    if reliable:
        interpretation = "supports_directional_hypothesis" if auc["lower_95"] > 0 else (
            "favors_selected_traditional_model" if auc["upper_95"] < 0 else "inconclusive")
    summary = {
        "comparison": f"cnn minus {comparator}", "primary_metric": "roc_auc",
        "positive_class": "high radiologist-rated suspicion", "unit": "nodule",
        "resampling_unit": "patient", "interval": "95% percentile patient-cluster bootstrap",
        "seed": seed, "requested_replicates": replicates, "valid_replicates": valid,
        "attempts": attempts, "rejected_single_class": rejected,
        "reliable": reliable, "interpretation": interpretation, "differences": difference_rows,
        "uncertainty_scope": "Test-patient sampling conditional on fixed models and thresholds; excludes training/model-selection variability",
    }
    return pd.DataFrame(rows), summary


def subgroup_metrics(predictions: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    data = validate_predictions(predictions)
    definitions = {}
    if "diameter_mm" in data:
        definitions["diameter_mm"] = pd.cut(data.diameter_mm, [-np.inf, 6, 10, np.inf],
                                            labels=["<=6", ">6 to 10", ">10"])
    if "rating_range" in data:
        definitions["rating_range"] = pd.cut(data.rating_range, [-np.inf, 1, np.inf], labels=["0-1", ">=2"])
    rows = []
    for variable, bins in definitions.items():
        temp = data.assign(_subgroup=bins)
        for (name, category), group in temp.groupby(["model", "_subgroup"], observed=True, sort=True):
            result = classification_metrics(group.label, group.score, group.probability, thresholds[name])
            result.update(model=name, variable=variable, subgroup=str(category), n_patients=group.patient_id.nunique(),
                          interpretation="descriptive only; no subgroup superiority inference")
            rows.append(result)
    return pd.DataFrame(rows)


def _plots(data: pd.DataFrame, thresholds: Mapping[str, float], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
    calibration_rows = []
    for name, group in data.groupby("model", sort=True):
        y, scores, probability = group.label.to_numpy(), group.score.to_numpy(), group.probability.to_numpy()
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, scores):.3f})")
        if (y == 1).any():
            precision, recall, _ = precision_recall_curve(y, scores)
            ax_pr.plot(recall, precision, label=f"{name} (AP {average_precision_score(y, scores):.3f})")
        order = np.argsort(probability, kind="stable")
        bins = np.array_split(order, min(5, len(order)))
        predicted, observed = [], []
        for index, members in enumerate(bins):
            predicted.append(float(probability[members].mean()))
            observed.append(float(y[members].mean()))
            calibration_rows.append({"model": name, "bin": index + 1, "n_nodules": len(members),
                                     "mean_probability": predicted[-1], "observed_fraction": observed[-1]})
        ax_cal.plot(predicted, observed, marker="o", label=name)
        decision = scores >= thresholds[name]
        counts = np.array([[(~decision & (y == 0)).sum(), (decision & (y == 0)).sum()],
                           [(~decision & (y == 1)).sum(), (decision & (y == 1)).sum()]])
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.imshow(counts, cmap="Blues")
        for (row, column), value in np.ndenumerate(counts):
            ax.text(column, row, str(value), ha="center", va="center", color="black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Low", "High"], yticklabels=["Low", "High"],
               xlabel="Predicted suspicion", ylabel="Reference suspicion", title=name)
        fig.tight_layout()
        fig.savefig(output / f"confusion_{name}.png", dpi=180)
        plt.close(fig)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="grey")
    prevalence = data.drop_duplicates("nodule_id").label.mean()
    ax_pr.axhline(prevalence, linestyle="--", color="grey", label=f"Prevalence {prevalence:.3f}")
    for figure, axis, filename, xlabel, ylabel in (
        (fig_roc, ax_roc, "roc.png", "False-positive rate", "Sensitivity"),
        (fig_pr, ax_pr, "precision_recall.png", "Recall", "Precision"),
        (fig_cal, ax_cal, "calibration.png", "Mean predicted high-suspicion probability", "Observed high-suspicion fraction"),
    ):
        axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.15)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)
    pd.DataFrame(calibration_rows).to_csv(output.parent / "calibration_bins.csv", index=False)


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def evaluate_predictions(predictions: pd.DataFrame, thresholds: Mapping[str, float], config: dict,
                         output_dir, comparator: str) -> dict:
    """Write reproducible test tables and figures from stored frozen predictions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = validate_predictions(predictions)
    if set(data.model) != {"logistic_regression", "svm", "random_forest", "cnn"}:
        raise ValueError("Final evaluation requires all four prespecified models")
    if "partition" in data and set(data.partition) != {"test"}:
        raise ValueError("Final evaluation accepts test predictions only")
    if data.label.nunique() != 2:
        raise ValueError("The frozen final test cohort must contain both reference classes")
    settings = config.get("evaluation", {})
    main = metrics_table(data, thresholds)
    probability = metrics_table(data, thresholds, probability_threshold=True)
    patient = metrics_table(data, thresholds, patient_weighted=True)
    intervals, primary = paired_patient_bootstrap(
        data, thresholds, comparator,
        replicates=int(settings.get("bootstrap_replicates", 2000)),
        max_attempts=int(settings.get("bootstrap_max_attempts", 20000)),
        seed=int(settings.get("bootstrap_seed", 42)),
    )
    data.to_csv(output / "test_predictions.csv", index=False)
    main.to_csv(output / "metrics.csv", index=False)
    probability.to_csv(output / "metrics_probability_050.csv", index=False)
    patient.to_csv(output / "metrics_patient_weighted.csv", index=False)
    intervals.to_csv(output / "metric_intervals.csv", index=False)
    subgroup_metrics(data, thresholds).to_csv(output / "subgroup_metrics.csv", index=False)
    with (output / "primary_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_value(primary), stream, indent=2, allow_nan=False)
    _plots(data, thresholds, output / "figures")
    return {"output_dir": str(output), "metrics": main.to_dict("records"), "primary_comparison": primary}
