"""Stages that keep development, freezing and final evaluation explicit."""
from __future__ import annotations

import json
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .splitting import allocation_summary, assign_partitions, assert_independent
from .utils import MODELS, config_hash, hash_file, read_csv_snapshot, write_json


def experiment_status(root):
    root = Path(root)
    data = {"output_dir": str(root.resolve()), "prepared_nodules": 0, "cohort_frozen": False,
            "selected_models": [], "final_evaluation_complete": False}
    manifest = root / "data" / "all_nodules.csv"
    if manifest.exists():
        rows = read_csv_snapshot(manifest)
        data["prepared_nodules"] = len(rows)
        data["prepared_patients"] = int(rows.patient_id.nunique()) if len(rows) else 0
    summary = root / "data" / "preparation_summary.json"
    if summary.exists():
        data["preparation_summary_matches_manifest"] = json.loads(summary.read_text())["prepared_nodules"] == data["prepared_nodules"]
    cohort = root / "cohort_summary.json"
    if cohort.exists():
        data["cohort_frozen"] = True
        data["cohort"] = json.loads(cohort.read_text())
    selection = root / "training" / "selection.json"
    if selection.exists():
        data["selected_models"] = list(json.loads(selection.read_text())["models"])
    data["final_evaluation_complete"] = (root / "results" / "primary_comparison.json").exists()
    data["protocol_frozen"] = (root / "frozen.json").exists()
    return data


def read_tables(root):
    root = Path(root)
    cohort = pd.read_csv(root / "cohort.csv")
    features = pd.read_csv(root / "features" / "radiomics_features.csv")
    assert_independent(cohort)
    if not set(cohort.nodule_id) <= set(features.nodule_id):
        raise ValueError("Missing feature rows in the common cohort")
    return cohort, features


def freeze_cohort(root):
    root = Path(root)
    preparation = json.loads((root / "data" / "preparation_summary.json").read_text())
    source_errors = pd.read_csv(root / "data" / "exclusions.csv")
    if source_errors.reason.isin(["RuntimeError", "FileNotFoundError", "PermissionError", "SourceUnavailableError"]).any():
        raise ValueError("Source-access failures remain in preparation; restore access and resume before freezing")
    prepared = pd.read_csv(root / "data" / "all_nodules.csv")
    if prepared.empty:
        raise ValueError("No prepared observations; inspect software and source errors before cohort construction")
    if len(prepared) != preparation["prepared_nodules"]:
        raise ValueError("Preparation is incomplete or its summary does not match the manifest")
    # Guard older in-progress preparation records against entering the primary
    # cohort before the explicit physical-size eligibility check was introduced.
    prepared = prepared[prepared.diameter_mm >= 3.0].copy()
    features = pd.read_csv(root / "features" / "radiomics_features.csv")
    if prepared.nodule_id.duplicated().any() or features.nodule_id.duplicated().any():
        raise ValueError("Duplicated identities in prepared images or extracted features")
    audit = pd.read_csv(root / "data" / "audit_nodules.csv")
    excluded_images = set(source_errors.loc[source_errors.stage.isin(["geometry", "preparation"]), "nodule_id"].dropna())
    if set(prepared.nodule_id) != set(audit.nodule_id) - excluded_images:
        raise ValueError("Prepared inputs do not reconcile with audited observations and recorded image exclusions")
    feature_errors = pd.read_csv(root / "features" / "feature_errors.csv")
    if not feature_errors.empty and not feature_errors.reason.eq("DataQualityError").all():
        raise ValueError("Unresolved software failures are present in feature extraction")
    successes, failures = set(features.nodule_id), set(feature_errors.nodule_id)
    if successes & failures or successes | failures != set(prepared.nodule_id):
        raise ValueError("Feature extraction is incomplete or does not reconcile with prepared inputs")
    common = prepared[prepared.nodule_id.isin(features.nodule_id)].copy()
    binary = common[common.label.isin([0, 1])].copy()
    cohort = assign_partitions(binary)
    target = root / "cohort.csv"
    if target.exists():
        old = pd.read_csv(target)
        columns = ["nodule_id", "patient_id", "label", "partition", "fold", "source_fingerprint"]
        pd.testing.assert_frame_equal(old[columns].reset_index(drop=True), cohort[columns].reset_index(drop=True), check_dtype=False)
    else:
        cohort.to_csv(target, index=False)
    common.to_csv(root / "common_all_labels.csv", index=False)
    source = json.loads((root / "data" / "audit_summary.json").read_text())
    flow = [{"stage": "source_collection", "patients": source["source_patients"],
             "series": source["source_scans"], "nodules": None}]
    for stage, rows in (("selected_series_reader_size_eligibility", audit), ("image_geometry_qc", prepared),
                        ("radiomics_qc", common), ("primary_binary", cohort)):
        flow.append({"stage": stage, "patients": int(rows.patient_id.nunique()),
                     "series": int(rows.series_uid.nunique()), "nodules": len(rows)})
    for name, rows in cohort.groupby("partition"):
        flow.append({"stage": name, "patients": int(rows.patient_id.nunique()),
                     "series": int(rows.series_uid.nunique()), "nodules": len(rows)})
    pd.DataFrame(flow).to_csv(root / "cohort_flow.csv", index=False)
    source_errors.groupby(["stage", "reason"]).size().rename("records").reset_index().assign(
        note="Series-stage records count series; reader/geometry/preparation records count nodules").to_csv(
            root / "exclusion_summary.csv", index=False)
    predictors = [name for name in features if name.startswith("original_")]
    values = features[predictors].to_numpy(dtype=float)
    write_json(root / "feature_qc.json", {"rows": len(features), "predictors": len(predictors),
               "families": pd.Series([name.split("_", 2)[1] for name in predictors]).value_counts().to_dict(),
               "nonfinite_cells": int((~np.isfinite(values)).sum()),
               "duplicate_feature_vectors": int(features[predictors].duplicated().sum()),
               "learned_feature_processing": "Imputation, constant/correlation filtering and scaling are fitted within training folds only"})
    write_json(root / "cohort_summary.json", {
        "prepared_nodules": len(prepared), "feature_success_nodules": len(common),
        "feature_excluded": len(prepared) - len(common), "mean_indeterminate": int((common.label == -1).sum()),
        "binary_nodules": len(cohort), "binary_patients": int(cohort.patient_id.nunique()),
        "partitions": allocation_summary(cohort), "manifest_sha256": hash_file(target),
        "endpoint": "low versus high radiologist-rated nodule malignancy suspicion"})
    return cohort


def model_paths(root):
    return json.loads((Path(root) / "training" / "model_artifacts.json").read_text())


def load_models(root):
    from .training import load_model
    paths = model_paths(root)
    if set(paths) != set(MODELS):
        raise ValueError("All four model artifacts are required")
    return {name: load_model(path) for name, path in paths.items()}


def make_predictions(models, cohort, features, config):
    from .evaluation import validate_predictions
    from .training import predict_model
    columns = [c for c in ("nodule_id", "patient_id", "label", "partition", "diameter_mm", "rating_range", "reader_count") if c in cohort]
    predictions = []
    for name, model in models.items():
        frame = predict_model(model, cohort, features, config)
        if len(frame) != len(cohort) or frame.nodule_id.duplicated().any() or set(frame.nodule_id) != set(cohort.nodule_id):
            raise ValueError(f"Incomplete requested prediction cohort for {name}")
        predictions.append(frame.merge(cohort[columns], on="nodule_id", validate="one_to_one").assign(model=name))
    return validate_predictions(pd.concat(predictions, ignore_index=True))


def select_thresholds(root, config):
    from .evaluation import choose_threshold, metrics_table
    root = Path(root)
    cohort, features = read_tables(root)
    validation = cohort[cohort.partition == "validation"]
    predictions = make_predictions(load_models(root), validation, features, config)
    target = float(config.get("evaluation", {}).get("target_sensitivity", 0.9))
    thresholds = {name: choose_threshold(rows.label, rows.score, target) for name, rows in predictions.groupby("model")}
    output = root / "development"
    output.mkdir(exist_ok=True)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    metrics_table(predictions, thresholds).to_csv(output / "validation_metrics.csv", index=False)
    write_json(output / "thresholds.json", thresholds)
    return thresholds


def freeze_experiment(root, config):
    """Seal exact models, thresholds, data and configuration before test scoring."""
    root = Path(root)
    cohort, features = read_tables(root)
    paths = model_paths(root)
    if set(paths) != set(MODELS):
        raise ValueError("Cannot freeze an incomplete model comparison")
    required = [root / "cohort.csv", root / "features" / "radiomics_features.csv",
                root / "training" / "selection.json", root / "development" / "thresholds.json",
                root / "secondary" / "complete.json", root / "explanations" / "explanation_summary.json",
                root / "checks" / "release_checks.json", root / "data" / "geometry_review.json",
                root / "environment.json", root / "requirements.lock.txt"]
    required += [root / "data" / name for name in ("all_nodules.csv", "audit_nodules.csv", "exclusions.csv", "preprocessing.json")
                 if (root / "data" / name).exists()]
    flagged_review = root / "data" / "flagged_case_review.json"
    if flagged_review.exists():
        required.append(flagged_review)
    corrections = root / "protocol_corrections.json"
    if corrections.exists():
        required.append(corrections)
    required += sorted((root / "data" / "source_panels").glob("*"))
    required += sorted((root / "environments").glob("*.json"))
    required += sorted(Path(__file__).parent.glob("*.py"))
    files = {str(p.resolve()): hash_file(p) for p in required}
    for path in paths.values():
        files[str(Path(path).resolve())] = hash_file(path)
        sidecar = Path(str(path) + ".metadata.json")
        files[str(sidecar.resolve())] = hash_file(sidecar)
    for path in cohort.npz_path.unique():
        files[str(Path(path).resolve())] = hash_file(path)
    release = json.loads((root / "checks" / "release_checks.json").read_text())
    if not release.get("passed"):
        raise ValueError("Release checks have not passed")
    frozen = {"config_hash": config_hash(config), "files": files,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "test_nodules": int((cohort.partition == "test").sum()),
              "primary_comparator": json.loads((root / "training" / "selection.json").read_text())["selected_traditional"]}
    packages = {key.lower(): value for key, value in json.loads((root / "environment.json").read_text())["packages"].items()}
    core = ("numpy", "pandas", "scikit-learn", "torch", "pyradiomics", "simpleitk", "pydicom", "pylidc")
    frozen["runtime_versions"] = {key: packages[key] for key in core if key in packages}
    marker = root / "frozen.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["files"] != files or old["config_hash"] != frozen["config_hash"]:
            raise ValueError("Frozen experiment has changed; create a new experiment")
        return old
    write_json(marker, frozen)
    return frozen


def verify_frozen(root, config):
    frozen = json.loads((Path(root) / "frozen.json").read_text())
    if frozen["config_hash"] != config_hash(config):
        raise ValueError("Configuration changed after freeze")
    for path, expected in frozen["files"].items():
        if hash_file(path) != expected:
            raise ValueError(f"Artifact changed after freeze: {path}")
    for name, expected in frozen.get("runtime_versions", {}).items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Runtime version changed after freeze: {name}")
    return frozen


def release_checks(root, config):
    from .training import load_model, predict_model
    root = Path(root)
    cohort, features = read_tables(root)
    review_path = root / "data" / "geometry_review.json"
    if not review_path.exists():
        raise ValueError("Inspect development_overlays.png, then record the review with the review stage")
    review = json.loads(review_path.read_text())
    if not review.get("reviewed") or review["overlay_sha256"] != hash_file(root / "data" / "development_overlays.png"):
        raise ValueError("Development geometry review is missing or refers to another overlay")
    exclusions = root / "data" / "exclusions.csv"
    if exclusions.exists() and review.get("exclusions_sha256") != hash_file(exclusions):
        raise ValueError("Data exclusions changed after quality review; inspect the updated cases and record the review again")
    for path, expected in review.get("source_panels", {}).items():
        if hash_file(path) != expected:
            raise ValueError("Source QA panels changed after visual review")
    validation = cohort[cohort.partition == "validation"].head(4)
    tolerances = {"rtol": 1e-6, "atol": 1e-7}
    results = {}
    for name, path in model_paths(root).items():
        model = load_model(path)
        expected = set(cohort.loc[cohort.partition == "train", "patient_id"])
        if set(model.metadata["fit_patient_ids"]) != expected:
            raise ValueError(f"Unexpected fitted patient provenance for {name}")
        expected_nodules = set(cohort.loc[cohort.partition == "train", "nodule_id"])
        if set(model.metadata["fit_nodule_ids"]) != expected_nodules:
            raise ValueError(f"Unexpected fitted nodule provenance for {name}")
        if name != "cnn":
            processor = model.feature_processor
            if set(processor.fit_patient_ids_) != expected or set(processor.fit_nodule_ids_) != expected_nodules:
                raise ValueError(f"Unexpected transformer fitting provenance for {name}")
        first = predict_model(model, validation, features, config)
        second = predict_model(load_model(path), validation, features, config)
        np.testing.assert_allclose(first[["score", "probability"]], second[["score", "probability"]], **tolerances)
        results[name] = {"reload_matches": True, "training_provenance_matches": True}
    result = {"passed": len(results) == 4, "models": results, "reload_tolerances": tolerances,
              "patient_and_series_overlap": 0,
              "visual_review": review}
    write_json(root / "checks" / "release_checks.json", result)
    return result


def record_geometry_review(root, note):
    root = Path(root)
    if not note or not note.strip():
        raise ValueError("Supply --review-note describing the inspected geometry and any unresolved issues")
    path = root / "data" / "development_overlays.png"
    result = {"reviewed": True, "note": note.strip(), "overlay_sha256": hash_file(path),
              "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    exclusions = root / "data" / "exclusions.csv"
    if exclusions.exists():
        result["exclusions_sha256"] = hash_file(exclusions)
    result["source_panels"] = {str(panel.resolve()): hash_file(panel)
                               for panel in sorted((root / "data" / "source_panels").glob("*.png"))}
    write_json(root / "data" / "geometry_review.json", result)
    return result


def development_overlays(root):
    """Deterministic development-only panels spanning size and reader counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    cohort, _ = read_tables(root)
    development = cohort[cohort.partition != "test"].copy()
    development["size_bin"] = pd.cut(development.diameter_mm, [0, 6, 10, np.inf], include_lowest=True)
    chosen = development.sort_values("nodule_id").groupby(["size_bin", "reader_count"], observed=True).head(1)
    if chosen.empty:
        raise ValueError("No development cases for geometry review")
    fig, axes = plt.subplots(int(np.ceil(len(chosen) / 4)), 4, figsize=(12, 3 * int(np.ceil(len(chosen) / 4))), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), chosen.itertuples()):
        with np.load(row.npz_path) as case:
            ax.imshow(case["hu"], cmap="gray", vmin=-1000, vmax=400)
            ax.contour(case["mask"], levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(f"{row.patient_id}\n{row.diameter_mm:.1f} mm; {row.reader_count} readers", fontsize=8)
    fig.tight_layout()
    target = root / "data" / "development_overlays.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    chosen.to_csv(root / "data" / "overlay_cases.csv", index=False)
    return target
