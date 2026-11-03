"""Comparable batch-one CPU costs, separated from disk and GPU training."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .utils import write_json


def model_only_predict(model, record, features, tensor=None, prepared_values=None):
    """Use prepared in-memory inputs; no NPZ I/O in the timed call."""
    import torch
    if model.model_name == "cnn":
        with torch.inference_mode():
            return float(torch.sigmoid(model.network(tensor)).item())
    data = prepared_values if prepared_values is not None else model.feature_processor.transform(features)
    if model.model_name == "svm":
        score = model.estimator.decision_function(data)
        probability = model.calibrator.predict(score)
        return float(score[0]), float(probability[0])
    return float(model.estimator.predict_proba(data)[0, 1])


def benchmark_models(root, config):
    import torch
    import SimpleITK as sitk
    from .dataset import _load_scan, _prepare_record
    from .features import build_extractor, extract_arrays
    from .workflow import load_models, read_tables, verify_frozen
    from .compat import load_pylidc
    pl = load_pylidc()

    root = Path(root)
    verify_frozen(root, config)
    cohort, features = read_tables(root)
    test = cohort[cohort.partition == "test"].sort_values("nodule_id")
    output = root / "resources"
    output.mkdir(exist_ok=True)
    models = load_models(root)
    models["cnn"].network.cpu().eval()
    torch.set_num_threads(int(config.get("runtime", {}).get("cpu_threads", 1)))
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(config.get("runtime", {}).get("cpu_threads", 1)))
    lookup = features.set_index("nodule_id")
    prepared = {}
    vectors = {}
    for row in test.itertuples():
        with np.load(row.npz_path) as case:
            prepared[row.nodule_id] = torch.from_numpy(case["image"].copy()).unsqueeze(0)
        vectors[row.nodule_id] = {name: model.feature_processor.transform(lookup.loc[[row.nodule_id]])
                                  for name, model in models.items() if name != "cnn"}
    timings = []
    with threadpool_limits(limits=int(config.get("runtime", {}).get("cpu_threads", 1))):
        first = test.iloc[0]
        x = lookup.loc[[first.nodule_id]]
        for name, model in models.items():
            for _ in range(10):
                model_only_predict(model, first, x, prepared[first.nodule_id], vectors[first.nodule_id].get(name))
        for repeat in range(5):
            for record in test.to_dict("records"):
                x = lookup.loc[[record["nodule_id"]]]
                for name, model in models.items():
                    start = time.perf_counter_ns()
                    model_only_predict(model, record, x, prepared[record["nodule_id"]], vectors[record["nodule_id"]].get(name))
                    timings.append({"model": name, "definition": "model_only", "repeat": repeat,
                                    "nodule_id": record["nodule_id"], "seconds": (time.perf_counter_ns() - start) / 1e9})
        # Full preprocessing starts with a decoded CT and supplied annotations.
        # One scan is held at a time. Each pass shares its independently timed
        # common transformation across model paths; disk decoding stays separate.
        extractor, _ = build_extractor(config)
        decode_times = []
        for repeat in range(5):
            for scan_id, group in test.groupby("scan_id", sort=True):
                scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).one()
                started = time.perf_counter()
                raw, geometry = _load_scan(scan, config)
                # Annotation retrieval belongs to input acquisition, before the
                # decoded-volume + supplied-contours pipeline clock begins.
                for annotation in scan.annotations:
                    tuple(annotation.contours)
                decode_times.append({"scan_id": int(scan_id), "repeat": repeat, "seconds": time.perf_counter() - started})
                for record in group.to_dict("records"):
                    started = time.perf_counter()
                    case = _prepare_record(record, config, None, scan, raw, geometry)["arrays"]
                    common_seconds = time.perf_counter() - started
                    tensor = torch.from_numpy(case["image"]).unsqueeze(0)
                    started = time.perf_counter()
                    values, _ = extract_arrays(case["hu"], case["mask"], extractor)
                    extraction_seconds = time.perf_counter() - started
                    current = pd.DataFrame([values])
                    for name, model in models.items():
                        started = time.perf_counter()
                        model_only_predict(model, record, current, tensor)
                        seconds = common_seconds + (time.perf_counter() - started)
                        if name != "cnn":
                            seconds += extraction_seconds
                        timings.append({"model": name, "definition": "decoded_ct_pipeline_uncached_features",
                                        "repeat": repeat, "nodule_id": record["nodule_id"], "seconds": seconds})
                del raw
            pd.DataFrame(timings).to_csv(output / "latency_samples.csv", index=False)
            print(f"CPU pipeline timing pass {repeat + 1}/5 complete: {len(test)} nodules", flush=True)
        pd.DataFrame(decode_times).to_csv(output / "source_decode_times.csv", index=False)
    samples = pd.DataFrame(timings)
    summary = samples.groupby(["model", "definition"]).seconds.agg(
        median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), samples="size").reset_index()
    summary["iqr"] = summary.q75 - summary.q25
    summary.to_csv(output / "latency_summary.csv", index=False)
    from .workflow import model_paths
    paths = model_paths(root)
    selection = json.loads((root / "training" / "selection.json").read_text())
    costs = [{"model": name, "search_wall_seconds": selection["search_wall_seconds"][name],
              "cv_fit_and_scoring_seconds": selection.get("cv_fit_and_scoring_seconds", {}).get(name),
              "final_fit_wall_seconds": model.metadata["fit_wall_seconds"],
              "fit_device": model.metadata.get("device", "cpu"),
              "inference_device": "cpu", "cpu_threads": config.get("runtime", {}).get("cpu_threads", 1),
              "saved_model_bytes": Path(paths[name]).stat().st_size,
              "fitted_feature_dimension": len(model.retained_features) if name != "cnn" else None,
              "peak_process_rss_bytes": model.metadata.get("peak_process_rss_bytes"),
              "peak_cuda_allocated_bytes": model.metadata.get("peak_cuda_allocated_bytes")}
             for name, model in models.items()]
    pd.DataFrame(costs).to_csv(output / "training_resources.csv", index=False)
    write_json(output / "measurement.json", {"cpu_threads": config.get("runtime", {}).get("cpu_threads", 1),
               "simpleitk_threads": sitk.ProcessObject_GetGlobalDefaultNumberOfThreads(),
               "batch_size": 1, "warmups": 10, "timed_passes": 5, "gpu_inference": False,
               "pipeline_disk_cache_note": "CT decode timed separately; all prepared inputs/features are generated in memory without disk cache",
               "pipeline_input": "decoded HU CT volume and supplied reader annotation contours",
               "source_decode_table": "includes CT file retrieval, integrity hashing, DICOM/HU decoding and annotation retrieval",
               "human_contouring_included": False})
    if not (output / "component_measurement.json").exists():
        from .component_benchmark import benchmark_components
        benchmark_components(root, config)
    return summary
