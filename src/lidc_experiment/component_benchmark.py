"""One supplementary pass that records the components of frozen CPU pipelines.

These samples supplement, and never replace, the original five-pass inference
benchmark.  All integrity comparisons run outside the component clocks.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .utils import MODELS, atomic_replace, hash_file, write_json


def _write_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    atomic_replace(temporary, path)


def benchmark_components(root, config):
    """Record one complete decoded-CT component pass without fitting any model."""
    import SimpleITK as sitk
    import torch
    from .benchmark import model_only_predict
    from .compat import load_pylidc
    from .dataset import _load_scan, _prepare_record
    from .features import build_extractor, extract_arrays
    from .workflow import load_models, read_tables, verify_frozen

    root = Path(root)
    frozen = verify_frozen(root, config)
    prediction_path = root / "results" / "test_predictions.csv"
    if not prediction_path.is_file():
        raise ValueError("Completed frozen-test predictions are required before component measurement")
    output = root / "resources"
    output.mkdir(exist_ok=True)
    complete_path = output / "component_measurement.json"
    if complete_path.exists():
        raise ValueError("Component measurement is already complete; refusing to overwrite its record")
    cohort, features = read_tables(root)
    test = cohort[cohort.partition == "test"].sort_values(["scan_id", "nodule_id"]).copy()
    if test.empty or test.nodule_id.duplicated().any():
        raise ValueError("Component measurement requires a nonempty unique frozen test list")
    predictions = pd.read_csv(prediction_path)
    if predictions.duplicated(["model", "nodule_id"]).any() or set(predictions.model) != set(MODELS):
        raise ValueError("Recorded test predictions must contain all four models without duplicated observations")
    for name in MODELS:
        if set(predictions.loc[predictions.model == name, "nodule_id"]) != set(test.nodule_id):
            raise ValueError(f"Recorded prediction cohort is incomplete for {name}")
    reference_scores = predictions.set_index(["model", "nodule_id"])["score"]
    feature_names = sorted(c for c in features if c.startswith("original_"))
    reference_features = features.set_index("nodule_id")
    models = load_models(root)
    models["cnn"].network.cpu().eval()
    torch.set_num_threads(1)
    previous_sitk_threads = sitk.ProcessObject_GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    for name, model in models.items():
        if name != "cnn" and getattr(model.estimator, "n_jobs", 1) not in (None, 1):
            raise ValueError(f"{name} has a parallel estimator; this supplement requires one CPU thread")
    pl = load_pylidc()
    logger = logging.getLogger("radiomics")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    samples, decode_times = [], []
    started_at = datetime.now(timezone.utc).isoformat()
    warmups = 10
    sample_path = output / "component_samples.csv"
    decode_path = output / "component_source_decode_times.csv"
    tolerance = {"feature_rtol": 1e-6, "feature_atol": 1e-8,
                 "ml_score_rtol": 1e-6, "ml_score_atol": 1e-7,
                 "cnn_score_rtol": 0.0, "cnn_score_atol": 1e-6}

    def prepare_components(record, scan, raw, geometry):
        started = time.perf_counter_ns()
        case = _prepare_record(record, config, None, scan, raw, geometry)["arrays"]
        common = (time.perf_counter_ns() - started) / 1e9
        started = time.perf_counter_ns()
        values, _ = extract_arrays(case["hu"], case["mask"], extractor)
        extraction = (time.perf_counter_ns() - started) / 1e9
        started = time.perf_counter_ns()
        current = pd.DataFrame([values])
        dataframe = (time.perf_counter_ns() - started) / 1e9
        started = time.perf_counter_ns()
        tensor = torch.from_numpy(case["image"]).unsqueeze(0)
        tensor_seconds = (time.perf_counter_ns() - started) / 1e9
        return case, values, current, tensor, {
            "common_preparation_seconds": common,
            "radiomics_extraction_seconds": extraction,
            "radiomics_dataframe_seconds": dataframe,
            "cnn_tensor_preparation_seconds": tensor_seconds,
        }

    try:
        with threadpool_limits(limits=1):
            extractor, parameters = build_extractor(config)
            for scan_number, (scan_id, group) in enumerate(test.groupby("scan_id", sort=True), 1):
                scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).one()
                started = time.perf_counter_ns()
                raw, geometry = _load_scan(scan, config)
                for annotation in scan.annotations:
                    tuple(annotation.contours)
                decode_times.append({"scan_id": int(scan_id), "seconds": (time.perf_counter_ns() - started) / 1e9})
                records = group.sort_values("nodule_id").to_dict("records")
                if not samples:
                    # Warm-up samples are deliberately discarded.  Each full
                    # path starts from this same already decoded first CT.
                    for _ in range(warmups):
                        _, _, current, tensor, _ = prepare_components(records[0], scan, raw, geometry)
                        for model in models.values():
                            model_only_predict(model, records[0], current, tensor)
                for record in records:
                    case, values, current, tensor, timings = prepare_components(record, scan, raw, geometry)
                    row = {"nodule_id": record["nodule_id"], "patient_id": record["patient_id"],
                           "scan_id": int(scan_id), "pass": 0, **timings}
                    native_scores = {}
                    for name, model in models.items():
                        started = time.perf_counter_ns()
                        result = model_only_predict(model, record, current, tensor)
                        seconds = (time.perf_counter_ns() - started) / 1e9
                        column = f"{name}_prediction_seconds" if name == "cnn" else f"{name}_transform_prediction_seconds"
                        row[column] = seconds
                        native_scores[name] = result[0] if name == "svm" else result
                        branch = (timings["cnn_tensor_preparation_seconds"] if name == "cnn" else
                                  timings["radiomics_extraction_seconds"] + timings["radiomics_dataframe_seconds"])
                        row[f"{name}_route_sum_seconds"] = timings["common_preparation_seconds"] + branch + seconds
                    # Frozen-array/feature/score verification is outside every
                    # timing region and cannot choose or refit a model.
                    with np.load(record["npz_path"], allow_pickle=False) as expected:
                        for key in ("hu", "mask", "image"):
                            np.testing.assert_array_equal(case[key], expected[key],
                                                          err_msg=f"Regenerated {key} differs for {record['nodule_id']}")
                    if set(values) != set(feature_names):
                        raise ValueError(f"Extracted feature names differ for {record['nodule_id']}")
                    np.testing.assert_allclose([values[c] for c in feature_names],
                                               reference_features.loc[record["nodule_id"], feature_names].to_numpy(dtype=float),
                                               rtol=tolerance["feature_rtol"], atol=tolerance["feature_atol"], equal_nan=True,
                                               err_msg=f"Regenerated features differ for {record['nodule_id']}")
                    for name, score in native_scores.items():
                        prefix = "cnn" if name == "cnn" else "ml"
                        np.testing.assert_allclose(score, reference_scores.loc[(name, record["nodule_id"])],
                                                   rtol=tolerance[f"{prefix}_score_rtol"],
                                                   atol=tolerance[f"{prefix}_score_atol"],
                                                   err_msg=f"Frozen native score differs for {name}/{record['nodule_id']}")
                    samples.append(row)
                del raw
                _write_csv(pd.DataFrame(samples), sample_path)
                _write_csv(pd.DataFrame(decode_times), decode_path)
                print(f"Component supplement: {scan_number}/{test.scan_id.nunique()} scans, "
                      f"{len(samples)}/{len(test)} nodules verified", flush=True)
        frame = pd.DataFrame(samples)
        timing_columns = [c for c in frame if c.endswith("_seconds")]
        durations = frame[timing_columns].to_numpy(dtype=float)
        if (len(frame) != len(test) or set(frame.nodule_id) != set(test.nodule_id)
                or frame.nodule_id.duplicated().any() or not np.isfinite(durations).all()
                or not np.all(durations > 0)):
            raise ValueError("Supplementary component samples are incomplete, nonfinite or nonpositive")
        summary = pd.DataFrame([{
            "component": name, "median": frame[name].median(), "q25": frame[name].quantile(.25),
            "q75": frame[name].quantile(.75), "iqr": frame[name].quantile(.75) - frame[name].quantile(.25),
            "samples": len(frame), "unit": "seconds",
        } for name in timing_columns])
        summary_path = output / "component_summary.csv"
        _write_csv(summary, summary_path)
        verify_frozen(root, config)
        write_json(complete_path, {
            "completed": True, "kind": "supplementary_component_timing", "timed_passes": 1,
            "test_nodules": len(test), "test_patients": int(test.patient_id.nunique()),
            "cpu_threads": 1, "simpleitk_threads": 1, "device": "cpu", "batch_size": 1,
            "ordering": "ascending scan_id, then ascending nodule_id; one complete frozen test-list pass",
            "ordered_nodule_ids": test.nodule_id.tolist(), "warmups": warmups,
            "warmup_scope": "ten untimed complete paths on the first case from decoded CT, including uncached features and all four model branches",
            "definitions": {
                "common_preparation_seconds": "complete _prepare_record: supplied reader-mask reconstruction, common resampling, consensus/slice/crop, HU clipping and two-channel image construction",
                "radiomics_extraction_seconds": "uncached extract_arrays on the regenerated selected HU slice/mask, including SimpleITK feature input creation and diagnostics",
                "radiomics_dataframe_seconds": "one-row numerical feature dataframe construction",
                "cnn_tensor_preparation_seconds": "torch.from_numpy plus batch dimension; two-channel array construction is already included in common preparation",
                "ml_transform_prediction_seconds": "fitted feature transform plus classifier prediction; SVM also applies its frozen sigmoid calibration",
                "cnn_prediction_seconds": "CPU eval/inference-mode network forward pass, sigmoid and scalar conversion",
                "route_sum_seconds": "common preparation plus the route's measured representation assembly and prediction components; a component sum, not an independent integrated wall-clock trial",
                "source_decode_seconds": "separate CT retrieval/integrity hashing, DICOM/HU decoding and supplied annotation retrieval; excluded from route sums",
            },
            "feature_cache": "uncached regeneration; no cached feature vector or prepared NPZ used as timed input",
            "common_work_shared": True, "human_contouring_included": False,
            "integrity_checks_outside_timers": True, "regenerated_arrays_identical": True,
            "regenerated_features_and_native_scores_match": True, "integrity_tolerances": tolerance,
            "model_refits": 0, "used_for_selection": False, "supplements_original_five_pass_benchmark": True,
            "frozen_config_hash": frozen["config_hash"], "frozen_manifest_sha256": hash_file(root / "frozen.json"),
            "source_sha256": hash_file(Path(__file__)), "test_predictions_sha256": hash_file(prediction_path),
            "extractor_parameters": parameters,
            "started_utc": started_at, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": {path.name: hash_file(path) for path in (sample_path, decode_path, summary_path)},
        })
        return summary
    finally:
        logger.setLevel(previous_level)
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(previous_sitk_threads)
