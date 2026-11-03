"""Explicit original-image, single-slice PyRadiomics extraction."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import DataQualityError, _hash, _write_csv, _write_json


FEATURE_CLASSES = ("shape2D", "firstorder", "glcm", "glrlm", "glszm")


def build_extractor(config: dict):
    from radiomics import featureextractor
    settings = {"bin_width": config.get("preprocessing", {}).get("bin_width", 25.0),
                **config.get("radiomics", {})}
    params = {
        "imageType": {"Original": {}},
        "featureClass": {name: [] for name in FEATURE_CLASSES},
        "setting": {"binWidth": float(settings.get("bin_width", 25.0)), "force2D": True,
                    "force2Ddimension": 0, "normalize": False, "additionalInfo": True,
                    "correctMask": False, "geometryTolerance": 1e-6,
                    "minimumROIDimensions": 2, "label": 1},
    }
    return featureextractor.RadiomicsFeatureExtractor(params), params


def extract_case(npz_path: str | Path, extractor, spacing_mm: float = 1.0) -> tuple[dict, dict]:
    with np.load(npz_path, allow_pickle=False) as data:
        hu, mask = data["hu"].astype(np.float32), data["mask"].astype(np.uint8)
    return extract_arrays(hu, mask, extractor, spacing_mm)


def extract_arrays(hu, mask, extractor, spacing_mm: float = 1.0) -> tuple[dict, dict]:
    """Extract from prepared in-memory arrays, including for pipeline timing."""
    import SimpleITK as sitk
    hu, mask = np.asarray(hu, dtype=np.float32), np.asarray(mask, dtype=np.uint8)
    if hu.shape != mask.shape or hu.ndim != 2 or not mask.any() or not np.isfinite(hu).all():
        raise DataQualityError("Invalid prepared case for radiomics")
    image = sitk.GetImageFromArray(hu[None])
    roi = sitk.GetImageFromArray(mask[None])
    image.SetSpacing((float(spacing_mm),) * 3)
    roi.CopyInformation(image)
    try:
        result = extractor.execute(image, roi)
    except ValueError as exc:
        case_invalid = ("mask only contains 1 segmented voxel", "mask has too few dimensions", "Size of the ROI is too small")
        if any(message in str(exc) for message in case_invalid):
            raise DataQualityError(str(exc)) from exc
        raise
    features, diagnostics = {}, {}
    for name, value in result.items():
        if name.startswith("original_"):
            family = name.split("_", 2)[1]
            if family not in FEATURE_CLASSES:
                raise RuntimeError(f"Unexpected feature family {family}")
            scalar = np.asarray(value)
            if scalar.size != 1:
                raise RuntimeError(f"Non-scalar feature {name}")
            features[name] = float(scalar.item())
        elif name.startswith("diagnostics_"):
            diagnostics[name] = value.tolist() if isinstance(value, np.ndarray) else value
    if not features or not any(name.startswith("original_shape2D_") for name in features):
        raise RuntimeError("No 2D shape features produced; check single-slice settings")
    return features, diagnostics


def extract_features(cohort: pd.DataFrame, config: dict, output_dir: str | Path) -> pd.DataFrame:
    """Extract deterministic features, recording case failures before cohort freeze."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import radiomics
    import SimpleITK as sitk
    radiomics.setVerbosity(logging.ERROR)
    log_path = (output / "extraction_warnings.log").resolve()
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
               for handler in radiomics.logger.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.WARNING)
        radiomics.logger.addHandler(handler)
    extractor, params = build_extractor(config)
    version = importlib.metadata.version("pyradiomics")
    versions = {"pyradiomics": version, "runtime_version": radiomics.__version__,
                "numpy": np.__version__, "simpleitk": sitk.Version_VersionString()}
    signature = _hash({"parameters": params, "versions": versions,
                       "preprocessing": config.get("preprocessing", {})})
    provenance_path = output / "feature_extraction.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise DataQualityError("Feature cache uses a different configuration or PyRadiomics version")
    _write_json({"signature": signature, "parameters": params, **versions}, provenance_path)
    rows, errors, diagnostics = [], [], []
    cache = output / "feature_cases"
    cache.mkdir(exist_ok=True)
    spacing = float(config.get("preprocessing", {}).get("spacing_mm", 1.0))
    for index, record in enumerate(cohort.sort_values("nodule_id").to_dict("records"), start=1):
        key = record["nodule_id"]
        path = Path(record["npz_path"])
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_path = cache / f"{key}.json"
        started = time.perf_counter()
        try:
            if cache_path.exists():
                saved = json.loads(cache_path.read_text(encoding="utf-8"))
                if saved["signature"] != signature or saved["image_hash"] != image_hash:
                    raise DataQualityError(f"Incompatible feature cache for {key}")
            else:
                values, info = extract_case(path, extractor, spacing)
                saved = {"signature": signature, "image_hash": image_hash, "features": values,
                         "diagnostics": info, "extraction_seconds": time.perf_counter() - started}
                # Diagnostic tuples and NumPy scalars are serialized separately
                # from numerical model predictors.
                temp = cache_path.with_suffix(".tmp")
                temp.write_text(json.dumps(saved, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=True), encoding="utf-8")
                os.replace(temp, cache_path)
            rows.append({"nodule_id": key, **saved["features"]})
            diagnostics.append({"nodule_id": key, "image_hash": image_hash, "extraction_seconds": saved["extraction_seconds"], **saved["diagnostics"]})
        except DataQualityError as exc:
            # Cache incompatibility is not an observation exclusion.
            if cache_path.exists():
                raise
            errors.append({"nodule_id": key, "patient_id": record.get("patient_id", ""), "reason": "DataQualityError", "detail": str(exc)})
        if index % 50 == 0:
            print(f"Radiomics {index}/{len(cohort)} cases: {len(rows)} successful, {len(errors)} exclusions", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["nodule_id"])
    _write_csv(frame, output / "radiomics_features.csv")
    _write_csv(pd.DataFrame(errors, columns=["nodule_id", "patient_id", "reason", "detail"]), output / "feature_errors.csv")
    temp = output / "feature_diagnostics.jsonl.tmp"
    temp.write_text("".join(json.dumps(row, default=str, allow_nan=True) + "\n" for row in diagnostics), encoding="utf-8")
    os.replace(temp, output / "feature_diagnostics.jsonl")
    return frame
