"""Traceable feature explanations and CNN localization for prepared nodules."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def grad_cam(network, image: np.ndarray) -> tuple[np.ndarray, float]:
    """Grad-CAM of the high-suspicion logit at the last convolutional layer."""
    import torch
    import torch.nn.functional as functional

    layers = [layer for layer in network.modules() if isinstance(layer, torch.nn.Conv2d)]
    if not layers:
        raise ValueError("Grad-CAM requires a convolutional network")
    captured = {}

    def capture(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.update(gradient=gradient))

    hook = layers[-1].register_forward_hook(capture)
    was_training = network.training
    network.eval()
    network.zero_grad(set_to_none=True)
    try:
        device = next(network.parameters()).device
        tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
        if tensor.ndim != 3 or tensor.shape[0] != 2:
            raise ValueError("Expected a two-channel prepared nodule tensor")
        with torch.enable_grad():
            logit = network(tensor.unsqueeze(0)).reshape(-1)[0]
            logit.backward()
            activation, gradient = captured["activation"], captured["gradient"]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
            result = cam[0, 0].detach().cpu().numpy().astype(np.float32)
            minimum, maximum = float(result.min()), float(result.max())
            result = (result - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(result)
            return result, float(logit.detach().cpu())
    finally:
        hook.remove()
        network.zero_grad(set_to_none=True)
        network.train(was_training)


def select_error_panel(predictions: pd.DataFrame, thresholds: dict, comparator: str,
                       limit: int = 5) -> pd.DataFrame:
    """Union of sorted-ID cases in each primary model's confusion categories."""
    from .evaluation import validate_predictions

    data = validate_predictions(predictions)
    selected = set()
    selections = []
    for name in ("cnn", comparator):
        group = data.loc[data.model == name].sort_values("nodule_id").copy()
        if group.empty:
            raise ValueError(f"Missing error-panel predictions for {name}")
        decision = group.score.to_numpy() >= thresholds[name]
        group["category"] = np.select(
            [(group.label == 1) & decision, (group.label == 0) & decision,
             (group.label == 0) & ~decision, (group.label == 1) & ~decision],
            ["TP", "FP", "TN", "FN"], default="invalid",
        )
        for category, members in group.groupby("category", sort=True):
            for key in members.head(limit).nodule_id:
                selected.add(str(key))
                selections.append({"nodule_id": str(key), "selected_by": name, "category": category,
                                   "category_available": len(members)})
    result = data.loc[data.nodule_id.isin(selected)].copy()
    reasons = pd.DataFrame(selections).groupby("nodule_id").apply(
        lambda group: "; ".join(f"{row.selected_by}:{row.category} (n={row.category_available})"
                                for row in group.itertuples()), include_groups=False,
    )
    result["selection_reason"] = result.nodule_id.map(reasons)
    return result.sort_values(["nodule_id", "model"]).reset_index(drop=True)


def _read_case(record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(str(record["npz_path"])) as archive:
        hu = archive["hu"].astype(np.float32)
        mask = archive["mask"].astype(bool)
        image = archive["image"].astype(np.float32)
    return hu, mask, image


def _save_case_panel(record, predictions, thresholds, cam, output: Path, random_cam=None,
                     metadata: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hu, mask, _ = _read_case(record)
    columns = 3 if random_cam is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4.5))
    for axis in axes:
        axis.imshow(hu, cmap="gray", vmin=-1000, vmax=400)
        if mask.any() and not mask.all():
            axis.contour(mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.7)
        axis.axis("off")
    axes[0].set_title("Prepared image and supplied contour")
    axes[1].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.48)
    axes[1].set_title("Trained CNN: high-suspicion logit")
    if random_cam is not None:
        axes[2].imshow(random_cam, cmap="inferno", vmin=0, vmax=1, alpha=0.48)
        axes[2].set_title("Randomly initialized CNN")
    score_text = []
    for row in predictions.itertuples():
        score_text.append(f"{row.model}: score {row.score:.3f}, threshold {thresholds[row.model]:.3f}")
    label = "high" if int(record["label"]) else "low"
    title = f"{record['nodule_id']} | reference suspicion: {label}\n" + " | ".join(score_text)
    fig.suptitle(title, fontsize=8, wrap=True)
    fig.text(0.5, 0.015, "Research visualization; activation maps do not establish diagnostic reasoning", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.88))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    details = {"nodule_id": str(record["nodule_id"]), "patient_id": str(record["patient_id"]),
               "reference_label": int(record["label"]), "target": "high-suspicion output logit",
               "layer": "last Conv2d before final pooling", "input_shape": [2, *list(hu.shape)],
               "predictions": predictions.to_dict("records"), **(metadata or {})}
    output.with_suffix(".json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")


def _feature_explanations(models, validation, features, config, output: Path) -> None:
    from .training import predict_model

    repeats = int(config.get("explanations", {}).get("permutation_repeats", 20))
    seed = int(config.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    rows = []
    for name in ("logistic_regression", "svm", "random_forest"):
        if name not in models:
            continue
        model = models[name]
        processor = model.feature_processor
        names = list(processor.retained_features_)
        if name == "logistic_regression":
            coefficient = np.asarray(model.estimator.coef_).reshape(-1)
            if len(coefficient) != len(names):
                raise ValueError("Coefficient and fitted feature-name lengths disagree")
            pd.DataFrame({"feature": names, "standardized_coefficient": coefficient,
                          "effect": "log odds per training standard deviation"}).sort_values(
                              "standardized_coefficient", key=abs, ascending=False).to_csv(
                                  output / "logistic_coefficients.csv", index=False)
        baseline = predict_model(model, validation, features, config)
        label = validation.set_index("nodule_id").loc[baseline.nodule_id, "label"].to_numpy()
        if len(np.unique(label)) != 2:
            raise ValueError("Permutation AUC requires both validation classes")
        baseline_auc = float(roc_auc_score(label, baseline.score))
        validation_ids = set(validation.nodule_id.astype(str))
        positions = np.flatnonzero(features.nodule_id.astype(str).isin(validation_ids).to_numpy())
        for feature in names:
            changes = []
            for repeat in range(repeats):
                perturbed = features.copy()
                values = perturbed.iloc[positions][feature].to_numpy(copy=True)
                perturbed.iloc[positions, perturbed.columns.get_loc(feature)] = rng.permutation(values)
                predicted = predict_model(model, validation, perturbed, config)
                y = validation.set_index("nodule_id").loc[predicted.nodule_id, "label"].to_numpy()
                changes.append(baseline_auc - float(roc_auc_score(y, predicted.score)))
            rows.append({"model": name, "feature": feature, "baseline_auc": baseline_auc,
                         "mean_auc_decrease": float(np.mean(changes)) if changes else np.nan,
                         "std_auc_decrease": float(np.std(changes, ddof=1)) if len(changes) > 1 else np.nan,
                         "repetitions": repeats, "seed": seed})
    pd.DataFrame(rows).to_csv(output / "permutation_importance.csv", index=False)


def generate_explanations(models: dict, cohort: pd.DataFrame, features: pd.DataFrame,
                          thresholds: dict, config: dict, output_dir, comparator: str,
                          include_test: bool = False) -> dict:
    """Produce development explanations and optionally a frozen-test error panel."""
    import torch
    from .training import predict_model

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = cohort.loc[cohort.partition == "validation"].copy()
    if validation.empty:
        raise ValueError("Explanations require the threshold-validation partition")
    if "cnn" not in models:
        raise ValueError("CNN model is required for activation-map explanations")
    _feature_explanations(models, validation, features, config, output)
    predictions = []
    for name, model in models.items():
        predicted = predict_model(model, validation, features, config)
        predicted = predicted.merge(validation[["patient_id", "nodule_id", "label"]], on="nodule_id", validate="one_to_one")
        predictions.append(predicted.assign(model=name))
    validation_predictions = pd.concat(predictions, ignore_index=True)
    network = models["cnn"].network
    device = next(network.parameters()).device
    cuda_devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(42)
        random_network = copy.deepcopy(network)
        for module in random_network.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    random_network.eval()
    panel_limit = int(config.get("explanations", {}).get("development_cases_per_class", 6))
    selected = validation.sort_values("nodule_id").groupby("label", group_keys=False).head(panel_limit)
    panel_dir = output / "development_panels"
    panel_dir.mkdir(exist_ok=True)
    sanity = []
    for _, record in selected.iterrows():
        _, _, image = _read_case(record)
        cam, logit = grad_cam(network, image)
        random_cam, random_logit = grad_cam(random_network, image)
        correlation = float(np.corrcoef(cam.ravel(), random_cam.ravel())[0, 1]) if cam.std() and random_cam.std() else np.nan
        row = {"nodule_id": record.nodule_id, "trained_logit": logit, "random_logit": random_logit,
               "map_mean_absolute_difference": float(np.abs(cam - random_cam).mean()),
               "map_pearson_correlation": correlation, "random_seed": 42,
               "trained_map_constant": bool(not cam.std()), "random_map_constant": bool(not random_cam.std())}
        sanity.append(row)
        case_predictions = validation_predictions.loc[validation_predictions.nodule_id == record.nodule_id]
        _save_case_panel(record, case_predictions, thresholds, cam, panel_dir / f"{record.nodule_id}.png", random_cam,
                         metadata={"partition": "validation", "model_metadata": models["cnn"].metadata})
    pd.DataFrame(sanity).to_csv(output / "parameter_randomization_sanity.csv", index=False)
    test_count = 0
    if include_test:
        test = cohort.loc[cohort.partition == "test"].copy()
        predictions = []
        for name, model in models.items():
            predicted = predict_model(model, test, features, config)
            predictions.append(predicted.merge(test[["patient_id", "nodule_id", "label"]], on="nodule_id", validate="one_to_one").assign(model=name))
        panel = select_error_panel(pd.concat(predictions, ignore_index=True), thresholds, comparator)
        panel.to_csv(output / "test_error_selection.csv", index=False)
        test_dir = output / "test_error_panels"
        test_dir.mkdir(exist_ok=True)
        for key in panel.nodule_id.unique():
            record = test.loc[test.nodule_id == key].iloc[0]
            _, _, image = _read_case(record)
            cam, _ = grad_cam(network, image)
            _save_case_panel(record, panel.loc[panel.nodule_id == key], thresholds, cam, test_dir / f"{key}.png",
                             metadata={"partition": "test", "model_metadata": models["cnn"].metadata})
            test_count += 1
    summary = {"development_cases": len(sanity), "test_cases": test_count,
               "permutation_repetitions": int(config.get("explanations", {}).get("permutation_repeats", 20)),
               "scope": "Traceability and learned-parameter dependence; no numerical interpretability superiority or clinical utility claim"}
    (output / "explanation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
