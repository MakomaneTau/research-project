"""Source-to-prepared-image panels for development-only visual quality review."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .compat import load_pylidc
from .dataset import DataQualityError, _annotation_masks, _load_scan
from .utils import atomic_replace, read_csv_snapshot


def _native_view(raw, annotations, center_xyz, field_mm):
    """A native-grid view at the nearest acquisition plane, without interpolation."""
    import SimpleITK as sitk

    center = np.asarray(raw.TransformPhysicalPointToContinuousIndex(tuple(center_xyz)), dtype=float)
    size = np.asarray(raw.GetSize(), dtype=int)
    spacing = np.asarray(raw.GetSpacing(), dtype=float)
    z_index = int(np.clip(np.floor(center[2] + 0.5), 0, size[2] - 1))
    low = np.maximum(0, np.floor(center[:2] - field_mm / (2 * spacing[:2]) - 1).astype(int))
    high = np.minimum(size[:2], np.ceil(center[:2] + field_mm / (2 * spacing[:2]) + 1).astype(int))
    if np.any(high <= low):
        raise DataQualityError("Prepared crop center falls outside its native CT series")
    region = sitk.RegionOfInterest(raw, [int(high[0] - low[0]), int(high[1] - low[1]), 1],
                                  [int(low[0]), int(low[1]), z_index])
    hu = sitk.GetArrayFromImage(region)[0]
    # imshow extents denote pixel edges, while contour coordinates denote centers.
    extent = ((low[0] - center[0] - 0.5) * spacing[0],
              (high[0] - center[0] - 0.5) * spacing[0],
              (high[1] - center[1] - 0.5) * spacing[1],
              (low[1] - center[1] - 0.5) * spacing[1])
    masks = _annotation_masks(None, annotations, raw)
    votes = None
    mask_x = mask_y = None
    for image in masks:
        offset = np.asarray(raw.TransformPhysicalPointToIndex(image.GetOrigin()), dtype=int)
        local_z = z_index - offset[2]
        if not 0 <= local_z < image.GetSize()[2]:
            continue
        plane = sitk.RegionOfInterest(image, [image.GetSize()[0], image.GetSize()[1], 1], [0, 0, int(local_z)])
        pixels = sitk.GetArrayFromImage(plane)[0].astype(np.uint16)
        if votes is None:
            votes = np.zeros_like(pixels, dtype=np.uint16)
            mask_x = (np.arange(pixels.shape[1]) + offset[0] - center[0]) * spacing[0]
            mask_y = (np.arange(pixels.shape[0]) + offset[1] - center[1]) * spacing[1]
        votes += pixels
    mask = None if votes is None else votes * 2 >= len(masks)
    native_center = raw.TransformContinuousIndexToPhysicalPoint((float(center[0]), float(center[1]), float(z_index)))
    return {"hu": hu, "extent": extent, "mask": mask, "mask_x": mask_x, "mask_y": mask_y,
            "native_slice_index": z_index, "native_center_xyz": native_center,
            "native_continuous_center_index": center.tolist(), "native_spacing_xyz": spacing.tolist()}


def source_overlay_panels(root, config) -> list[Path]:
    """Render selected development cases against their original acquisition CT.

    The selection is supplied by ``data/overlay_cases.csv``. This function never
    selects cases using model outputs or accesses a row explicitly marked test.
    Native images show the nearest acquired slice, so their plane may differ
    from the linearly interpolated standardized image. Panels support a human
    review; generating them alone does not constitute clinical adjudication.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    selection = read_csv_snapshot(root / "data" / "overlay_cases.csv")
    required = {"nodule_id", "patient_id", "scan_id", "annotation_ids", "npz_path"}
    if not required.issubset(selection.columns):
        raise ValueError(f"Overlay selection is missing {sorted(required - set(selection.columns))}")
    if "partition" in selection and not selection.partition.isin(["train", "validation"]).all():
        raise ValueError("Source QA panels require development patients only")
    if selection.nodule_id.duplicated().any():
        raise ValueError("Duplicate nodule in source QA selection")
    output = root / "data" / "source_panels"
    output.mkdir(parents=True, exist_ok=True)
    pl = load_pylidc()
    records = []
    paths = []
    settings = config.get("preprocessing", {})
    hu_min, hu_max = float(settings.get("hu_min", -1000)), float(settings.get("hu_max", 400))
    for scan_id, group in selection.groupby("scan_id", sort=True):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == int(scan_id)).one()
        raw, geometry = _load_scan(scan, config)
        for record in group.sort_values("nodule_id").to_dict("records"):
            if record["patient_id"] != scan.patient_id:
                raise DataQualityError("Selected overlay patient does not match its scan")
            with np.load(record["npz_path"], allow_pickle=False) as archive:
                hu = archive["hu"].astype(np.float32)
                mask = archive["mask"].astype(bool)
                channels = archive["image"].astype(np.float32)
                metadata = json.loads(str(archive["metadata"].item()))
            if metadata["nodule_id"] != record["nodule_id"]:
                raise DataQualityError("Prepared overlay archive has the wrong nodule identity")
            if hu.shape != mask.shape or channels.shape != (2, *hu.shape) or not mask.any():
                raise DataQualityError("Prepared overlay arrays are inconsistent")
            if not np.allclose(metadata["native_direction"], raw.GetDirection(), atol=1e-6):
                raise DataQualityError("Native direction differs from prepared image provenance")
            if json.loads(metadata["source_image_uids"]) != json.loads(geometry["source_image_uids"]):
                raise DataQualityError("Native source slices differ from prepared image provenance")
            if metadata.get("source_image_sha256") != geometry.get("source_image_sha256"):
                raise DataQualityError("Native source file checksums differ from prepared image provenance")
            ids = sorted(json.loads(record["annotation_ids"]))
            annotations = sorted([a for a in scan.annotations if a.id in ids], key=lambda a: a.id)
            if [a.id for a in annotations] != ids:
                raise DataQualityError("Selected overlay annotations do not belong to the scan")
            spacing = float(metadata["spacing_mm"])
            if hu.shape[0] != hu.shape[1]:
                raise DataQualityError("Source QA expects a square prepared field of view")
            field_mm = hu.shape[0] * spacing
            direction = np.asarray(metadata["native_direction"]).reshape(3, 3)
            center_xyz = np.asarray(metadata["crop_origin_xyz"]) + direction @ np.array([
                (hu.shape[1] - 1) * spacing / 2, (hu.shape[0] - 1) * spacing / 2, 0])
            native = _native_view(raw, annotations, center_xyz, field_mm)
            label = {0: "low suspicion", 1: "high suspicion", -1: "indeterminate"}.get(record.get("label"), "not shown")
            fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8), constrained_layout=True)
            try:
                axes[0].imshow(native["hu"], cmap="gray", vmin=hu_min, vmax=hu_max,
                               extent=native["extent"], origin="upper", interpolation="nearest")
                if native["mask"] is not None and native["mask"].any():
                    axes[0].contour(native["mask_x"], native["mask_y"], native["mask"], levels=[0.5],
                                    colors=["#ffb84d"], linewidths=1)
                axes[0].set_title(f"Native CT: nearest acquired plane\nindex {native['native_slice_index']}, center z={native['native_center_xyz'][2]:.2f} mm")
                extent = (-field_mm / 2, field_mm / 2, field_mm / 2, -field_mm / 2)
                coords = (np.arange(hu.shape[0]) - (hu.shape[0] - 1) / 2) * spacing
                axes[1].imshow(hu, cmap="gray", vmin=hu_min, vmax=hu_max, extent=extent, origin="upper", interpolation="nearest")
                axes[1].contour(coords, coords, mask, levels=[0.5], colors=["#ffb84d"], linewidths=1)
                axes[1].set_title(f"Standardized HU + consensus\nindex {metadata['selected_slice_index']}, center z={center_xyz[2]:.2f} mm")
                axes[2].imshow(channels[0], cmap="gray", vmin=0, vmax=1, extent=extent, origin="upper", interpolation="nearest")
                axes[2].set_title("CNN channel 1\nmasked intensity [0, 1]")
                axes[3].imshow(channels[1], cmap="gray", vmin=0, vmax=1, extent=extent, origin="upper", interpolation="nearest")
                axes[3].set_title("CNN channel 2\nbinary consensus mask")
                for axis in axes:
                    axis.set_xlim(-field_mm / 2, field_mm / 2)
                    axis.set_ylim(field_mm / 2, -field_mm / 2)
                    axis.set_aspect("equal")
                    axis.set_xlabel("image-plane x offset (mm)")
                    axis.set_ylabel("image-plane y offset (mm)")
                    axis.set_facecolor("black")
                fig.suptitle(f"{record['patient_id']} | {record['nodule_id']} | {label}\n"
                             "Native nearest plane may differ from the interpolated standardized plane; supplied reader contours are inputs.", fontsize=11)
                destination = output / f"{record['nodule_id']}.png"
                temporary = destination.with_suffix(".tmp.png")
                fig.savefig(temporary, dpi=150)
                atomic_replace(temporary, destination)
                paths.append(destination)
            finally:
                plt.close(fig)
            records.append({"patient_id": record["patient_id"], "nodule_id": record["nodule_id"],
                            "scan_id": scan_id, "partition": record.get("partition", "selection_supplied_without_partition"),
                            "source_panel_path": str(destination.resolve()), "npz_path": str(record["npz_path"]),
                            "native_slice_index": native["native_slice_index"],
                            "standardized_slice_index": int(metadata["selected_slice_index"]),
                            "native_center_z_mm": native["native_center_xyz"][2],
                            "standardized_center_z_mm": float(center_xyz[2]),
                            "plane_center_separation_mm": float(np.linalg.norm(np.asarray(native["native_center_xyz"]) - center_xyz)),
                            "field_of_view_mm": field_mm, "native_consensus_visible": native["mask"] is not None and bool(native["mask"].any()),
                            "review_status": "generated_for_visual_review; not_expert_adjudicated"})
        del raw
    metadata_path = output / "selected_cases.csv"
    temporary = metadata_path.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    atomic_replace(temporary, metadata_path)
    return paths
