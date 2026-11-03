"""A results index generated exclusively from recorded experimental artifacts."""
import json
from pathlib import Path

import pandas as pd


def write_report(root):
    root = Path(root)
    required = [root / "results" / "metrics.csv", root / "results" / "primary_comparison.json"]
    for path in required:
        if not path.exists():
            raise ValueError("Final predictions and analysis must exist before writing the results report")
    cohort = json.loads((root / "cohort_summary.json").read_text())
    metrics = pd.read_csv(required[0])
    comparison = json.loads(required[1].read_text())
    def csv_table(frame):
        columns = list(frame.columns)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)
    rows = ["# Chapter 4 experimental evidence", "",
            "Endpoint: low versus high radiologist-rated malignancy suspicion in pre-annotated LIDC-IDRI nodules.", "",
            f"Common cohort: {cohort['binary_nodules']} nodules from {cohort['binary_patients']} patients.", "",
            "## Frozen-test model comparison", "", csv_table(metrics), "",
            "## Primary paired comparison", "", "```json", json.dumps(comparison, indent=2), "```", "",
            "## Evidence files", "",
            "- `cohort_summary.json`: cohort and partition denominators.",
            "- `training/cv_results.csv`: all candidate/fold scores and failures.",
            "- `results/test_predictions.csv`: traceable per-nodule predictions.",
            "- `results/metric_intervals.csv`: patient-bootstrap uncertainty.",
            "- `results/metrics_patient_weighted.csv` and `subgroup_metrics.csv`: secondary test summaries.",
            "- `secondary/development_metrics.csv`: learning, weighting, augmentation, label and seed analyses.",
            "- `secondary/mask_sensitivity.csv`: alternative-mask results and invalid alternatives.",
            "- `explanations/`: coefficients, permutation importance and parameter-randomization checks.",
            "- `test_explanations/test_error_panels/`: deterministic shared test-case review.",
            "- `resources/`: measured computational costs.", "",
            "The comparison estimates performance on the frozen internal cohort. Reader ratings are proxy labels; "
            "the experiment does not establish pathology-confirmed diagnosis, full-scan detection or clinical transfer.", ""]
    target = root / "CHAPTER4_RESULTS.md"
    target.write_text("\n".join(rows), encoding="utf-8")
    return str(target)
