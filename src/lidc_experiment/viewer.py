"""Static local viewer for prepared, annotated cases only."""
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


def view_case(root, config, nodule_id, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .workflow import load_models, make_predictions, read_tables, verify_frozen
    if not nodule_id:
        raise ValueError("Supply --nodule-id for a prepared annotated case; arbitrary full CT inputs are unsupported")
    root = Path(root)
    cohort, features = read_tables(root)
    case = cohort[cohort.nodule_id == nodule_id]
    if len(case) != 1:
        raise ValueError("Unknown prepared nodule; use an identifier from cohort.csv")
    if case.iloc[0].partition == "test" and not (root / "results" / "test_predictions.csv").exists():
        raise ValueError("Test-case predictions are unavailable until frozen final evaluation completes")
    if case.iloc[0].partition == "test":
        verify_frozen(root, config)
    thresholds = json.loads((root / "development" / "thresholds.json").read_text())
    predictions = make_predictions(load_models(root), case, features, config)
    predictions["threshold"] = predictions.model.map(thresholds)
    predictions["decision"] = np.where(predictions.score >= predictions.threshold, "high suspicion", "low suspicion")
    with np.load(case.iloc[0].npz_path) as arrays:
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(arrays["hu"], cmap="gray", vmin=-1000, vmax=400)
        axis.contour(arrays["mask"], levels=[.5], colors="cyan", linewidths=.8)
        axis.axis("off")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    image = base64.b64encode(buffer.getvalue()).decode()
    title = html.escape(nodule_id)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Prepared LIDC nodule</title>
<style>body{{font:16px system-ui;max-width:980px;margin:40px auto;padding:0 20px;color:#17212a}}img{{max-width:100%}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}small{{color:#555}}</style>
<h1>Prepared nodule: {title}</h1><p>Research classification of radiologist-rated suspicion. Supplied expert contours are required.
This system has no clinical validation and does not perform full-scan nodule detection.</p><img alt="CT nodule and supplied contour" src="data:image/png;base64,{image}">
{predictions[["model","probability","score","threshold","decision"]].to_html(index=False, float_format=lambda x:f"{x:.4f}")}
<p><small>Configuration and model identifiers are recorded in the experiment artifact directory.</small></p></html>'''
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target.resolve())
