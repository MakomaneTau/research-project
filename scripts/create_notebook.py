"""Build the executable artifact-inspection notebook without embedded results."""
from pathlib import Path

import nbformat as n


cells = [
    n.v4.new_markdown_cell("""# LIDC-IDRI: Chapter 3 experiment

Reproducible comparison of low/high radiologist-rated suspicion. This notebook reads
recorded experimental outputs. Preprocessing success is not predictive-performance evidence."""),
    n.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / "configs").exists():
    ROOT = ROOT.parent
ARTIFACTS = ROOT / "artifacts/chapter3"
print("Artifact directory:", ARTIFACTS)
status = ARTIFACTS / "status.json"
print(json.loads(status.read_text()) if status.exists() else "No CLI stage completion recorded yet.")'''),
    n.v4.new_markdown_cell("""## Run the staged experiment

From the repository root: `python -m lidc_experiment run --config configs/chapter3.yaml`.
Replace `run` with a stage to resume: `prepare`, `features`, `split`, `tune`, `fit`,
`thresholds`, `secondary`, `explain`, `check`, `freeze`, `evaluate`, `benchmark`, `report`.
Inspect development overlays before freezing. Limits and reduced grids require a separate smoke profile."""),
    n.v4.new_code_cell('''def table(relative):
    path = ARTIFACTS / relative
    if path.exists():
        result = pd.read_csv(path)
        display(result)
        return result
    print("Not produced yet:", relative)

summary = ARTIFACTS / "cohort_summary.json"
if summary.exists():
    display(json.loads(summary.read_text()))
else:
    print("Final common cohort is not frozen yet.")
table("training/cv_results.csv")'''),
    n.v4.new_code_cell('''table("results/metrics.csv")
table("results/metric_intervals.csv")
comparison = ARTIFACTS / "results/primary_comparison.json"
if comparison.exists():
    display(json.loads(comparison.read_text()))'''),
    n.v4.new_code_cell('''table("secondary/development_metrics.csv")
table("secondary/mask_sensitivity.csv")
table("resources/latency_summary.csv")
table("resources/training_resources.csv")
table("resources/component_summary.csv")'''),
    n.v4.new_code_cell('''for path in sorted((ARTIFACTS / "results/figures").glob("*.png")):
    display(Image(filename=str(path)))
learning = ARTIFACTS / "secondary/learning_curves.png"
if learning.exists():
    display(Image(filename=str(learning)))'''),
    n.v4.new_markdown_cell("""## Prepared-case viewer and interpretation

Use `python -m lidc_experiment view --nodule-id ID` with an identifier from `cohort.csv`.
It generates a local HTML view containing the supplied contour, model scores and thresholds.
Full CT examinations are unsupported; test-case predictions require completed final evaluation.

Reference labels represent radiologist suspicion, not pathology-confirmed cancer.
A confidence interval spanning zero is inconclusive and does not establish equivalence.
Read the [interpretation notes](../artifacts/chapter3/CHAPTER4_INTERPRETATION.md)
for the primary comparison, calibration limitations, explanation checks and exact timing boundaries."""),
]

book = n.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (LIDC experiment)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
target = Path(__file__).resolve().parents[1] / "notebooks" / "chapter3_experiment.ipynb"
target.parent.mkdir(exist_ok=True)
n.write(book, target)
print(target)
