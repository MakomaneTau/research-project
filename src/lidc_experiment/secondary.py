"""Prespecified development analyses; none can promote a test-selected model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .splitting import nested_patient_subsets
from .utils import MODELS, write_json


def median_development_cohort(all_labels, primary):
    """Alternative labels can add lesions, but never add held-out patients."""
    assignments = primary[primary.partition != "test"][["patient_id", "partition", "fold"]].drop_duplicates()
    alternative = all_labels.drop(columns=["partition", "fold"], errors="ignore").merge(
        assignments, on="patient_id", validate="many_to_one")
    alternative["primary_mean_label"] = alternative.label
    values = alternative.rating_median.to_numpy()
    alternative["label"] = np.where(values <= 2, 0, np.where(values >= 4, 1, -1))
    if set(alternative.patient_id) & set(primary.loc[primary.partition == "test", "patient_id"]):
        raise ValueError("Test patients entered alternative-label development")
    return alternative


def run_secondary(root, config):
    from .dataset import DataQualityError, prepare_case
    from .features import build_extractor, extract_case
    from .training import fit_selected_model, predict_model
    from .workflow import load_models, read_tables

    root = Path(root)
    output = root / "secondary"
    output.mkdir(exist_ok=True)
    cohort, features = read_tables(root)
    train, validation = (cohort[cohort.partition == p].copy() for p in ("train", "validation"))
    selection = json.loads((root / "training" / "selection.json").read_text())
    selected = selection["models"]
    rows = []

    def fit_analysis(analysis, name, subset, held, *, settings=None, seed=42, augment=True, detail=""):
        spec = selected[name]
        setting = copy.deepcopy(spec["settings"] if settings is None else settings)
        folder = output / "fits" / analysis / detail / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Secondary {analysis}/{detail}/{name}: {subset.patient_id.nunique()} training patients", flush=True)
        # SVM alternatives need native scores only. Avoid unnecessary refitting
        # of calibration folds when the analysis endpoint is development AUC/AP.
        model = fit_selected_model(name, subset, features, setting, config,
                                   folder / ("model.pt" if name == "cnn" else "model.joblib"),
                                   seed=seed, augment=augment, epochs=spec.get("epochs"), calibrate=False)
        prediction = predict_model(model, held, features, config).merge(
            held[["nodule_id", "patient_id", "label"]], on="nodule_id", validate="one_to_one")
        prediction.to_csv(folder / "validation_predictions.csv", index=False)
        record = {"analysis": analysis, "detail": detail, "model": name, "seed": seed,
                  "training_patients": int(subset.patient_id.nunique()), "training_nodules": len(subset),
                  "validation_patients": int(held.patient_id.nunique()), "validation_nodules": len(held),
                  "roc_auc": float(roc_auc_score(prediction.label, prediction.score)),
                  "average_precision": float(average_precision_score(prediction.label, prediction.score)),
                  "fit_wall_seconds": model.metadata["fit_wall_seconds"], "settings": json.dumps(setting, sort_keys=True)}
        rows.append(record)
        pd.DataFrame(rows).to_csv(output / "development_metrics.csv", index=False)

    fractions = config.get("secondary", {}).get("learning_fractions", [0.25, 0.5, 0.75, 1.0])
    for fraction, subset in nested_patient_subsets(train, fractions).items():
        for name in MODELS:
            fit_analysis("learning_curve", name, subset, validation, detail=f"fraction_{fraction:.2f}")
    for name in MODELS:
        setting = copy.deepcopy(selected[name]["settings"])
        if name == "cnn":
            setting["weighted_loss"] = not setting["weighted_loss"]
        else:
            setting["class_weight"] = "balanced" if setting.get("class_weight") is None else None
        fit_analysis("class_weight_toggle", name, train, validation, settings=setting)
    fit_analysis("no_augmentation", "cnn", train, validation, augment=False)
    for seed in config.get("secondary", {}).get("seeds", [17, 73]):
        for name in ("random_forest", "cnn"):
            fit_analysis("seed_stability", name, train, validation, seed=int(seed), detail=f"seed_{seed}")

    alternative = median_development_cohort(pd.read_csv(root / "common_all_labels.csv"), cohort)
    alternative.to_csv(output / "median_label_changes.csv", index=False)
    usable = alternative[alternative.label.isin([0, 1])]
    for name in MODELS:
        fit_analysis("median_labels", name, usable[usable.partition == "train"],
                     usable[usable.partition == "validation"])

    models = load_models(root)
    extractor, _ = build_extractor(config)
    count = int(config.get("secondary", {}).get("mask_cases_per_cell", 5))
    sample = validation.sort_values("nodule_id").groupby(["label", "reader_count"], group_keys=False).head(count)
    sample.to_csv(output / "mask_sample.csv", index=False)
    mask_rows = []
    for _, record in sample.iterrows():
        baseline_features = features.set_index("nodule_id").loc[record.nodule_id]
        baseline = {name: predict_model(model, record.to_frame().T, features, config).score.iloc[0]
                    for name, model in models.items()}
        for variant, index in [("strict", None)] + [("individual", i) for i in range(int(record.reader_count))]:
            key = f"{record.nodule_id}_{variant}_{index if index is not None else 'all'}"
            path = output / "masks" / f"{key}.npz"
            base = {"nodule_id": record.nodule_id, "variant": variant, "reader_index": index}
            try:
                changed = prepare_case(record, config, path, mask_rule="strict" if variant == "strict" else "half", reader_index=index)
                values, _ = extract_case(path, extractor)
                replacement = pd.DataFrame([{"nodule_id": record.nodule_id, **values}])
                case = pd.DataFrame([changed])
                feature_changes = []
                for feature, value in values.items():
                    original = float(baseline_features[feature])
                    feature_changes.append({**base, "feature": feature, "original_value": original,
                                            "alternative_value": value, "absolute_change": abs(value - original)})
                pd.DataFrame(feature_changes).to_csv(output / "masks" / f"{key}_features.csv", index=False)
                for name, model in models.items():
                    prediction = predict_model(model, case, replacement, config)
                    mask_rows.append({**base, "model": name, "status": "success",
                                      "original_slice": record.selected_slice_index,
                                      "alternative_slice": changed["selected_slice_index"],
                                      "original_score": baseline[name], "alternative_score": prediction.score.iloc[0],
                                      "score_change": prediction.score.iloc[0] - baseline[name]})
            except DataQualityError as exc:
                mask_rows.append({**base, "status": "invalid_alternative", "error": f"{type(exc).__name__}: {exc}"})
            pd.DataFrame(mask_rows).to_csv(output / "mask_sensitivity.csv", index=False)
    metrics = pd.DataFrame(rows)
    _plot_learning_curves(metrics, output)
    summary = {"completed": True, "development_fits": len(rows), "mask_cases": len(sample),
               "mask_prediction_rows": len(mask_rows), "test_used_for_selection": False,
               "median_eligible_nodules": len(usable),
               "median_newly_eligible": int(((alternative.primary_mean_label == -1) & alternative.label.isin([0, 1])).sum()),
               "median_changed_classes": int((alternative.primary_mean_label.isin([0, 1]) & alternative.label.isin([0, 1]) &
                                               (alternative.primary_mean_label != alternative.label)).sum())}
    write_json(output / "complete.json", summary)
    return summary


def _plot_learning_curves(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 5))
    for name, rows in metrics[metrics.analysis == "learning_curve"].groupby("model"):
        rows = rows.sort_values("training_patients")
        axis.plot(rows.training_patients, rows.roc_auc, marker="o", label=name)
    axis.set(xlabel="Training patients", ylabel="Threshold-validation ROC-AUC", ylim=(0, 1))
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
