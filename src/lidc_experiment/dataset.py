"""LIDC cohort audit and physically matched, single-slice model inputs.

Images use NumPy Z,Y,X and SimpleITK X,Y,Z coordinates.  The patient and
annotation keys are retained for provenance, never supplied as predictors.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compat import load_pylidc
from .utils import atomic_replace


class DataQualityError(ValueError):
    """A source observation cannot satisfy the frozen spatial protocol."""


class SourceUnavailableError(RuntimeError):
    """Source storage/access failed; this is not a scientific case exclusion."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    atomic_replace(temp, path)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    atomic_replace(temp, path)


def _settings(config: dict) -> dict:
    return {
        "spacing_mm": 1.0, "hu_min": -1000.0, "hu_max": 400.0,
        "crop_size": 64, "max_slice_spacing_mm": 2.5, "min_readers": 2,
        **config.get("preprocessing", {}),
    }


def preprocessing_hash(config: dict) -> str:
    return _hash({"protocol_version": 1, "settings": _settings(config)})


def suspicion_label(ratings: list[float] | np.ndarray, aggregation: str = "mean") -> int:
    ratings = np.asarray(ratings, dtype=float)
    if not len(ratings) or not np.isfinite(ratings).all() or np.any((ratings < 1) | (ratings > 5)):
        raise DataQualityError("Malignancy ratings must be finite values from 1 to 5")
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be mean or median")
    value = float(np.mean(ratings) if aggregation == "mean" else np.median(ratings))
    return 0 if value <= 2 else 1 if value >= 4 else -1


def consensus_mask(reader_masks: np.ndarray, rule: str = "half", reader_index: int | None = None) -> np.ndarray:
    masks = np.asarray(reader_masks, dtype=bool)
    if masks.ndim != 4 or not masks.shape[0]:
        raise DataQualityError("Reader masks must have shape R,Z,Y,X")
    if reader_index is not None:
        if not 0 <= reader_index < len(masks):
            raise IndexError("Reader index is outside the available masks")
        return masks[reader_index]
    votes = masks.sum(axis=0)
    if rule == "half":
        return votes * 2 >= len(masks)
    if rule == "strict":
        return votes * 2 > len(masks)
    raise ValueError("mask_rule must be half or strict")


def select_slice(mask_zyx: np.ndarray, z_spacing: float = 1.0) -> int:
    """Maximum mask area, then physical centroid distance, then lower index."""
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise DataQualityError("Cannot select a slice from an empty/non-3D mask")
    areas = mask.sum(axis=(1, 2))
    centroid_z = float(np.average(np.arange(len(areas)), weights=areas))
    candidates = np.flatnonzero(areas == areas.max())
    return int(min(candidates, key=lambda z: (abs(float(z) - centroid_z) * z_spacing, int(z))))


def crop_slice(hu_yx: np.ndarray, mask_yx: np.ndarray, crop_size: int = 64,
               hu_min: float = -1000, hu_max: float = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Center by mask centroid; preserve every lesion pixel, with fixed padding."""
    hu = np.asarray(hu_yx, dtype=np.float32)
    mask = np.asarray(mask_yx, dtype=bool)
    if hu.shape != mask.shape or hu.ndim != 2 or not mask.any() or not np.isfinite(hu).all():
        raise DataQualityError("Invalid image/mask plane")
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)
    start = np.floor(centroid - (crop_size - 1) / 2).astype(int)
    stop = start + crop_size
    if np.any(coords.min(axis=0) < start) or np.any(coords.max(axis=0) >= stop):
        raise DataQualityError("Selected lesion would be truncated by the fixed crop")
    out = np.full((crop_size, crop_size), hu_min, dtype=np.float32)
    out_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    lo = np.maximum(start, 0)
    hi = np.minimum(stop, hu.shape)
    dest_lo = lo - start
    dest_hi = dest_lo + hi - lo
    src = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(dest_lo, dest_hi))
    out[dst] = np.clip(hu[src], hu_min, hu_max)
    out_mask[dst] = mask[src]
    if out_mask.sum() != mask.sum():
        raise DataQualityError("Mask mapping lost lesion pixels")
    image = np.stack((((out - hu_min) / (hu_max - hu_min)) * out_mask, out_mask)).astype(np.float32)
    return out, out_mask, image, (int(start[0]), int(start[1]))


def cluster_annotations(scan) -> tuple[list[list], dict]:
    """Match pylidc's clustering, retaining resolvable groups after its exception.

    We compute the final adjacency ourselves because pylidc raises without
    returning the useful groups when one connected component remains >4.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    annotations = sorted(scan.annotations, key=lambda a: a.id)
    if len(annotations) <= 1:
        return ([annotations] if annotations else []), {"tolerance_mm": float(scan.slice_thickness), "ambiguous_clusters": 0}
    distances = np.zeros((len(annotations), len(annotations)), dtype=float)
    # pylidc's default min metric is min(cdist(contours_matrix_A,
    # contours_matrix_B)). Exact nearest-neighbor queries compute the same
    # distance without repeatedly materializing contour arrays or huge N*M
    # distance matrices. eps=0 keeps this exact, not approximate clustering.
    points = [np.asarray(annotation.contours_matrix, dtype=float) for annotation in annotations]
    trees = [cKDTree(array) for array in points]
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            distances[i, j] = distances[j, i] = float(trees[i].query(points[j], k=1, eps=0)[0].min())
    tolerance = float(scan.slice_thickness)
    while True:
        count, labels = connected_components(distances <= tolerance, directed=False)
        groups = [[annotations[i] for i in range(len(annotations)) if labels[i] == g] for g in range(count)]
        if max(map(len, groups)) <= 4 or tolerance * 0.9 < 0.1:
            break
        tolerance *= 0.9
    groups.sort(key=lambda g: (float(np.mean([a.centroid[2] for a in g])), min(a.id for a in g)))
    return groups, {"tolerance_mm": tolerance, "ambiguous_clusters": sum(len(g) > 4 for g in groups)}


def _scan_dir(scan, config: dict) -> Path:
    root = Path(config["data_root"]).resolve()
    if not root.is_dir():
        raise SourceUnavailableError(f"Configured raw-data root is unavailable: {root}")
    try:
        path = Path(scan.get_path_to_dicom_files()).resolve()
    except (OSError, RuntimeError) as exc:
        raise SourceUnavailableError(f"Cannot access source CT for {scan.patient_id}; restore data access and resume") from exc
    if not path.is_relative_to(root):
        raise DataQualityError(f"pylidc resolves {path} outside configured data_root {root}")
    return path


def _source_fingerprint(scan, config: dict) -> str:
    root = _scan_dir(scan, config)
    files = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file())
    annotations = []
    for annotation in sorted(scan.annotations, key=lambda a: a.id):
        annotations.append({"id": annotation.id, "rating": annotation.malignancy,
                            "contours": [(c.id, c.image_z_position, c.inclusion, c.coords) for c in annotation.contours]})
    return _hash({"series": scan.series_instance_uid, "files": files, "annotations": annotations})


def audit_dataset(config: dict, output_dir: str | Path, limit_patients: int | None = None) -> pd.DataFrame:
    """Enumerate a deterministic cohort from the annotation database, without pixels.

    Metadata eligibility is provisional until prepare_dataset checks actual DICOM
    geometry and reconstructs each image/mask. No existing manifest is trusted.
    """
    pl = load_pylidc()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = _settings(config)
    min_readers = int(settings["min_readers"])
    scans = list(pl.query(pl.Scan).all())
    by_patient: dict[str, list] = {}
    for scan in scans:
        by_patient.setdefault(scan.patient_id, []).append(scan)
    rows, exclusions = [], []
    patient_keys = sorted(by_patient)
    if limit_patients is not None:
        if limit_patients <= 0:
            raise ValueError("limit_patients must be positive")
        patient_keys = patient_keys[:limit_patients]
    for patient_number, patient_id in enumerate(patient_keys, start=1):
        candidates = []
        for scan in by_patient[patient_id]:
            spacing = float(scan.slice_spacing)
            if not math.isfinite(spacing) or spacing <= 0 or spacing > float(settings["max_slice_spacing_mm"]):
                exclusions.append({"patient_id": patient_id, "scan_id": scan.id, "nodule_id": "", "stage": "series", "reason": "through_plane_spacing", "detail": str(spacing)})
                continue
            # Series with no multi-reader nodule cannot contribute to this study.
            try:
                groups, clustering = cluster_annotations(scan)
                useful = [g for g in groups if min_readers <= len(g) <= 4]
                if not useful:
                    exclusions.append({"patient_id": patient_id, "scan_id": scan.id, "nodule_id": "", "stage": "series", "reason": "no_eligible_reader_group", "detail": ""})
                    continue
                candidates.append((spacing, str(scan.series_instance_uid), scan, groups, clustering))
            except (AttributeError, ImportError, TypeError):
                raise
            except Exception as exc:
                exclusions.append({"patient_id": patient_id, "scan_id": scan.id, "nodule_id": "", "stage": "clustering", "reason": type(exc).__name__, "detail": str(exc)})
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[:2])
        _, _, selected, groups, clustering = candidates[0]
        for _, _, other, _, _ in candidates[1:]:
            exclusions.append({"patient_id": patient_id, "scan_id": other.id, "nodule_id": "", "stage": "series", "reason": "additional_series", "detail": str(selected.series_instance_uid)})
        try:
            fingerprint = _source_fingerprint(selected, config)
        except (SourceUnavailableError, OSError):
            raise
        except Exception as exc:
            exclusions.append({"patient_id": patient_id, "scan_id": selected.id, "nodule_id": "", "stage": "source", "reason": type(exc).__name__, "detail": str(exc)})
            continue
        for annotations in groups:
            ids = sorted(int(a.id) for a in annotations)
            key = "n_" + _hash({"series": selected.series_instance_uid, "annotations": ids})[:20]
            base = {"patient_id": patient_id, "scan_id": int(selected.id), "nodule_id": key}
            if len(annotations) > 4 or len(annotations) < min_readers:
                exclusions.append({**base, "stage": "readers", "reason": "ambiguous_cluster" if len(annotations) > 4 else "insufficient_readers", "detail": json.dumps(ids)})
                continue
            ratings = [float(a.malignancy) for a in annotations]
            try:
                label = suspicion_label(ratings)
            except DataQualityError as exc:
                exclusions.append({**base, "stage": "ratings", "reason": "invalid_ratings", "detail": str(exc)})
                continue
            diameter_mm = float(np.mean([a.diameter for a in annotations]))
            if not math.isfinite(diameter_mm) or diameter_mm < 3.0:
                exclusions.append({**base, "stage": "size", "reason": "diameter_below_3mm", "detail": str(diameter_mm)})
                continue
            rows.append({**base, "study_uid": str(selected.study_instance_uid), "series_uid": str(selected.series_instance_uid),
                         "annotation_ids": json.dumps(ids), "ratings": json.dumps(ratings),
                         "rating_mean": float(np.mean(ratings)), "rating_median": float(np.median(ratings)),
                         "rating_range": float(np.ptp(ratings)), "reader_count": len(ratings),
                         "diameter_mm": diameter_mm,
                         "spacing_z_mm": float(selected.slice_spacing), "label": label,
                         "clustering_tolerance_mm": clustering["tolerance_mm"],
                         "source_fingerprint": fingerprint, "preprocessing_hash": preprocessing_hash(config),
                         "npz_path": str((output / "images" / f"{key}.npz").resolve())})
        if patient_number % 25 == 0:
            print(f"Audited {patient_number}/{len(patient_keys)} patients: {len(rows)} candidate nodules", flush=True)
    columns = ["patient_id", "scan_id", "nodule_id", "study_uid", "series_uid", "annotation_ids", "ratings", "rating_mean", "rating_median", "rating_range", "reader_count", "diameter_mm", "spacing_z_mm", "label", "clustering_tolerance_mm", "source_fingerprint", "preprocessing_hash", "npz_path"]
    frame = pd.DataFrame(rows, columns=columns)
    _write_csv(frame, output / "audit_nodules.csv")
    _write_csv(pd.DataFrame(exclusions, columns=["patient_id", "scan_id", "nodule_id", "stage", "reason", "detail"]), output / "audit_exclusions.csv")
    _write_json({"source_patients": len(by_patient), "source_scans": len(scans), "limit_patients": limit_patients,
                 "audited_nodules": len(frame), "preprocessing_hash": preprocessing_hash(config)}, output / "audit_summary.json")
    return frame


def _load_scan(scan, config: dict):
    """Decode matching CT files and verify their physical sampling and identities."""
    import pydicom
    import SimpleITK as sitk
    root = _scan_dir(scan, config)
    headers = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".dcm":
            continue
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if str(getattr(ds, "Modality", "")) != "CT":
            continue
        if str(getattr(ds, "SeriesInstanceUID", "")) != str(scan.series_instance_uid) or str(getattr(ds, "StudyInstanceUID", "")) != str(scan.study_instance_uid):
            continue
        if str(getattr(ds, "PatientID", "")) != str(scan.patient_id):
            raise DataQualityError("DICOM patient identity does not match the annotation database")
        required = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns",
                    "SOPInstanceUID", "RescaleSlope", "RescaleIntercept")
        if any(not hasattr(ds, field) for field in required):
            raise DataQualityError("CT slice lacks required physical geometry or HU scaling fields")
        headers.append((ds, path))
    if len(headers) < 2:
        raise DataQualityError("Fewer than two matching CT slices")
    # Annotation k coordinates in pylidc are sorted by physical patient z.
    # Its duplicate-z convention chooses the lesser InstanceNumber.
    headers.sort(key=lambda pair: (float(pair[0].ImagePositionPatient[2]), int(getattr(pair[0], "InstanceNumber", 0)), str(pair[0].SOPInstanceUID)))
    unique = {}
    for ds, path in headers:
        unique.setdefault(round(float(ds.ImagePositionPatient[2]), 5), (ds, path))
    headers = list(unique.values())
    positions = np.asarray([ds.ImagePositionPatient for ds, _ in headers], dtype=float)
    if len(positions) != len(scan.slice_zvals) or not np.allclose(positions[:, 2], scan.slice_zvals, atol=1e-3, rtol=0):
        raise DataQualityError("Source slice positions do not match pylidc annotation indexing")
    first = headers[0][0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    x, y = orientation[:3], orientation[3:]
    if not np.allclose([np.linalg.norm(x), np.linalg.norm(y), np.dot(x, y)], [1, 1, 0], atol=1e-4):
        raise DataQualityError("Non-orthonormal DICOM orientation")
    normal = np.cross(x, y)
    deltas = np.diff(positions, axis=0)
    signed = deltas @ normal
    direction_z = normal * (1 if np.median(signed) > 0 else -1)
    spacing_z = float(np.median(deltas @ direction_z))
    if spacing_z <= 0 or spacing_z > float(_settings(config)["max_slice_spacing_mm"]) + 1e-4:
        raise DataQualityError("Actual through-plane spacing exceeds eligibility limit")
    # Irregular spacing and gantry shear must not be represented by a false affine.
    if not np.allclose(deltas, spacing_z * direction_z, atol=max(1e-3, spacing_z * 0.01), rtol=0):
        raise DataQualityError("Irregular slice positions or gantry shear")
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    if not np.isfinite(pixel_spacing).all() or np.any(pixel_spacing <= 0):
        raise DataQualityError("Invalid PixelSpacing")
    shape = (int(first.Rows), int(first.Columns))
    arrays = []
    uids = []
    checksums = {}
    for ds, path in headers:
        if (int(ds.Rows), int(ds.Columns)) != shape or not np.allclose(ds.ImageOrientationPatient, orientation, atol=1e-4) or not np.allclose(ds.PixelSpacing, pixel_spacing, atol=1e-5):
            raise DataQualityError("Inconsistent slice geometry")
        source_bytes = path.read_bytes()
        decoded = pydicom.dcmread(io.BytesIO(source_bytes))
        checksums[str(ds.SOPInstanceUID)] = hashlib.sha256(source_bytes).hexdigest()
        slope = float(getattr(decoded, "RescaleSlope", 1.0))
        intercept = float(getattr(decoded, "RescaleIntercept", 0.0))
        if not np.isfinite([slope, intercept]).all() or slope == 0:
            raise DataQualityError("Invalid HU rescale slope/intercept")
        pixels = decoded.pixel_array
        if pixels.ndim != 2 or pixels.shape != shape:
            raise DataQualityError("Unsupported multi-frame CT or inconsistent pixel dimensions")
        arrays.append(pixels.astype(np.float32) * slope + intercept)
        uids.append(str(ds.SOPInstanceUID))
    array = np.stack(arrays)
    if not np.isfinite(array).all():
        raise DataQualityError("Non-finite HU intensities")
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(tuple(positions[0]))
    image.SetSpacing((float(pixel_spacing[1]), float(pixel_spacing[0]), spacing_z))
    image.SetDirection(tuple(np.column_stack((x, y, direction_z)).ravel()))
    return image, {"source_image_uids": json.dumps(uids), "source_image_sha256": checksums,
                   "native_spacing_xyz": list(image.GetSpacing()), "native_origin_xyz": list(image.GetOrigin()), "native_direction": list(image.GetDirection())}


def _annotation_masks(scan, annotations, reference):
    import SimpleITK as sitk
    from pylidc.utils import consensus
    _, bbox_yxz, reader_masks_yxz = consensus(annotations, clevel=0.5)
    # Common annotation masks occupy a cropped native grid. Their physical origin
    # follows the crop offset; no full-sized mask array is needed.
    start_xyz = (int(bbox_yxz[1].start), int(bbox_yxz[0].start), int(bbox_yxz[2].start))
    origin = reference.TransformIndexToPhysicalPoint(start_xyz)
    images = []
    for mask in reader_masks_yxz:
        image = sitk.GetImageFromArray(np.transpose(mask, (2, 0, 1)).astype(np.uint8))
        image.SetSpacing(reference.GetSpacing())
        image.SetDirection(reference.GetDirection())
        image.SetOrigin(origin)
        images.append(image)
    return images


def _prepare_record(record: dict, config: dict, output_path: Path | None, scan, raw_image, geometry: dict,
                    mask_rule: str = "half", reader_index: int | None = None) -> dict:
    import SimpleITK as sitk
    settings = _settings(config)
    annotation_ids = json.loads(str(record["annotation_ids"]))
    annotations = [a for a in scan.annotations if int(a.id) in annotation_ids]
    if sorted(a.id for a in annotations) != sorted(annotation_ids):
        raise DataQualityError("Annotation IDs do not belong to their declared series")
    annotations.sort(key=lambda a: a.id)
    # Verify annotation slice references against the exact source series before
    # reconstructing masks. pylidc records z and original filename, not SOP UID.
    zs = np.asarray(scan.slice_zvals)
    for annotation in annotations:
        for contour in annotation.contours:
            if not np.isclose(zs, float(contour.image_z_position), atol=1e-3, rtol=0).any():
                raise DataQualityError("Annotation references a missing slice position")
    masks = _annotation_masks(scan, annotations, raw_image)
    spacing = float(settings["spacing_mm"])
    size = [int(math.ceil((n - 1) * s / spacing)) + 1 for n, s in zip(raw_image.GetSize(), raw_image.GetSpacing())]
    # Only the reference affine is needed; allocating the complete standardized
    # CT for each lesion would waste hundreds of megabytes.
    reference = sitk.Image([1, 1, 1], sitk.sitkFloat32)
    reference.SetSpacing((spacing,) * 3)
    reference.SetDirection(raw_image.GetDirection())
    reference.SetOrigin(raw_image.GetOrigin())
    # Resample only a local, reference-grid-aligned bounding box around the nodule.
    # This is mathematically the same transform as full-volume resampling, avoids
    # retaining hundreds of megabytes per lesion, and keeps output slice indices
    # traceable to the full standardized CT reference.
    corners = []
    for mask_image in masks:
        for z in (0, mask_image.GetSize()[2] - 1):
            for y in (0, mask_image.GetSize()[1] - 1):
                for x in (0, mask_image.GetSize()[0] - 1):
                    point = mask_image.TransformIndexToPhysicalPoint((x, y, z))
                    corners.append(reference.TransformPhysicalPointToContinuousIndex(point))
    corners = np.asarray(corners)
    margin = np.array([int(settings["crop_size"]), int(settings["crop_size"]), 2])
    low = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - margin)
    high = np.minimum(size, np.ceil(corners.max(axis=0)).astype(int) + margin + 1)
    local = sitk.Image([int(v) for v in high - low], sitk.sitkFloat32)
    local.SetSpacing(reference.GetSpacing())
    local.SetDirection(reference.GetDirection())
    local.SetOrigin(reference.TransformIndexToPhysicalPoint(tuple(int(v) for v in low)))
    transformed = []
    for mask_image in masks:
        resampled = sitk.Resample(mask_image, local, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        transformed.append(sitk.GetArrayFromImage(resampled).astype(bool))
    mask3d = consensus_mask(np.stack(transformed), mask_rule, reader_index)
    selected = select_slice(mask3d, spacing)
    # Sample intensities directly to the selected plane of the identical local
    # isotropic reference, using linear interpolation exactly once.
    plane = sitk.Image([local.GetSize()[0], local.GetSize()[1], 1], sitk.sitkFloat32)
    plane.SetSpacing(local.GetSpacing())
    plane.SetDirection(local.GetDirection())
    plane.SetOrigin(local.TransformIndexToPhysicalPoint((0, 0, selected)))
    sampled = sitk.Resample(raw_image, plane, sitk.Transform(), sitk.sitkLinear, float(settings["hu_min"]), sitk.sitkFloat32)
    hu, mask, image, crop_start_yx = crop_slice(sitk.GetArrayFromImage(sampled)[0], mask3d[selected],
                                              int(settings["crop_size"]), float(settings["hu_min"]), float(settings["hu_max"]))
    metadata = {"preprocessing_hash": preprocessing_hash(config), "source_fingerprint": record["source_fingerprint"],
                "nodule_id": record["nodule_id"], "mask_rule": mask_rule, "reader_index": reader_index,
                "selected_slice_index": int(selected + low[2]), "crop_start_yx": list(crop_start_yx),
                "crop_origin_xyz": list(plane.TransformIndexToPhysicalPoint((crop_start_yx[1], crop_start_yx[0], 0))),
                "spacing_mm": spacing, **geometry}
    if output_path is None:
        return {**record, "arrays": {"hu": hu, "mask": mask, "image": image}, "metadata": metadata,
                "selected_slice_index": metadata["selected_slice_index"],
                "lesion_area_mm2": int(mask.sum()) * spacing ** 2}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, hu=hu, mask=mask, image=image, metadata=np.array(json.dumps(metadata)))
    atomic_replace(temp, output_path)
    return {**record, "npz_path": str(output_path.resolve()), "selected_slice_index": metadata["selected_slice_index"],
            "lesion_area_mm2": int(mask.sum()) * spacing ** 2, "source_image_uids": geometry["source_image_uids"]}


def prepare_case(record: dict | pd.Series, config: dict, output_path: str | Path,
                 mask_rule: str = "half", reader_index: int | None = None) -> dict:
    """Prepare an original or alternative mask observation from canonical raw CT."""
    pl = load_pylidc()
    record = dict(record)
    scan = pl.query(pl.Scan).filter(pl.Scan.id == int(record["scan_id"])).one()
    actual_source = _source_fingerprint(scan, config)
    if actual_source != record["source_fingerprint"]:
        raise DataQualityError("Source files/annotations changed since the cohort audit")
    raw, geometry = _load_scan(scan, config)
    return _prepare_record(record, config, Path(output_path), scan, raw, geometry, mask_rule, reader_index)


def _cached_record(record: dict, config: dict) -> dict | None:
    path = Path(record["npz_path"])
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata["preprocessing_hash"] != preprocessing_hash(config) or metadata["source_fingerprint"] != record["source_fingerprint"] or metadata["nodule_id"] != record["nodule_id"]:
                raise DataQualityError(f"Incompatible cached case: {path}; use a fresh output directory")
            if data["image"].shape != (2, int(_settings(config)["crop_size"]), int(_settings(config)["crop_size"])) or not np.isfinite(data["image"]).all() or not data["mask"].any():
                raise DataQualityError(f"Invalid cached arrays: {path}")
            return {**record, "selected_slice_index": metadata["selected_slice_index"],
                    "lesion_area_mm2": int(data["mask"].sum()) * float(metadata["spacing_mm"]) ** 2,
                    "source_image_uids": metadata["source_image_uids"]}
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
        raise DataQualityError(f"Cannot resume cached case {path}: {exc}") from exc


def prepare_dataset(config: dict, output_dir: str | Path, limit_patients: int | None = None) -> pd.DataFrame:
    """Audit and prepare all eligible observations, retaining indeterminates too."""
    pl = load_pylidc()
    output = Path(output_dir)
    fingerprint = preprocessing_hash(config)
    protocol_path = output / "preprocessing.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8"))["preprocessing_hash"] != fingerprint:
        raise DataQualityError("Output directory contains another preprocessing configuration")
    output.mkdir(parents=True, exist_ok=True)
    _write_json({"preprocessing_hash": fingerprint, "settings": _settings(config)}, protocol_path)
    audited = audit_dataset(config, output, limit_patients=limit_patients)
    exclusions = pd.read_csv(output / "audit_exclusions.csv").to_dict("records")
    prepared = []
    start = time.perf_counter()
    for scan_id, group in audited.groupby("scan_id", sort=False):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).one()
        records = group.to_dict("records")
        pending = []
        for record in records:
            cached = _cached_record(record, config)
            if cached is None:
                pending.append(record)
            else:
                prepared.append(cached)
        if pending:
            try:
                raw, geometry = _load_scan(scan, config)
            except (AttributeError, ImportError, TypeError, SourceUnavailableError, OSError):
                raise
            except Exception as exc:
                for record in pending:
                    exclusions.append({"patient_id": record["patient_id"], "scan_id": scan_id, "nodule_id": record["nodule_id"], "stage": "geometry", "reason": type(exc).__name__, "detail": str(exc)})
                continue
            for record in pending:
                try:
                    prepared.append(_prepare_record(record, config, Path(record["npz_path"]), scan, raw, geometry))
                except (AttributeError, ImportError, TypeError, OSError):
                    raise
                except Exception as exc:
                    exclusions.append({"patient_id": record["patient_id"], "scan_id": scan_id, "nodule_id": record["nodule_id"], "stage": "preparation", "reason": type(exc).__name__, "detail": str(exc)})
            del raw
        frame = pd.DataFrame(prepared)
        _write_csv(frame, output / "all_nodules.csv")
        _write_csv(pd.DataFrame(exclusions), output / "exclusions.csv")
        print(f"Prepared scan {scan_id}: {len(prepared)} nodules cached", flush=True)
    frame = pd.DataFrame(prepared, columns=list(audited.columns) + ["selected_slice_index", "lesion_area_mm2", "source_image_uids"])
    _write_csv(frame, output / "all_nodules.csv")
    _write_csv(pd.DataFrame(exclusions), output / "exclusions.csv")
    _write_json({"preprocessing_hash": fingerprint, "limit_patients": limit_patients,
                 "prepared_nodules": len(frame), "prepared_patients": int(frame.patient_id.nunique()),
                 "labels": {str(k): int(v) for k, v in frame.label.value_counts().items()},
                 "elapsed_seconds": time.perf_counter() - start}, output / "preparation_summary.json")
    return frame
