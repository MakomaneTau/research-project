"""Run separately auditable stages, or the complete Chapter 3 experiment."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .utils import initialize_run, load_config, read_csv_snapshot, write_json


def run_stages(root):
    if (Path(root) / "frozen.json").exists():
        return [stage for stage in ("evaluate", "benchmark", "report")
                if not (Path(root) / "stages" / f"{stage}.json").exists()]
    return ["prepare", "features", "split", "tune", "fit", "thresholds", "secondary", "explain",
            "check", "freeze", "evaluate", "benchmark", "report"]


def execute(stage, root, config, args):
    from . import workflow
    if stage == "audit":
        from .dataset import audit_dataset
        return audit_dataset(config, root / "data", limit_patients=args.limit_patients)
    if stage == "prepare":
        from .dataset import prepare_dataset
        return prepare_dataset(config, root / "data", limit_patients=args.limit_patients)
    if stage == "features":
        from .features import extract_features
        return extract_features(read_csv_snapshot(root / "data" / "all_nodules.csv"), config, root / "features")
    if stage == "split":
        result = workflow.freeze_cohort(root)
        workflow.development_overlays(root)
        from .qa import source_overlay_panels
        source_overlay_panels(root, config)
        return result
    if stage == "tune":
        from .training import tune_models
        cohort, features = workflow.read_tables(root)
        return tune_models(cohort, features, config, root / "training")
    if stage == "fit":
        from .training import fit_final_models
        cohort, features = workflow.read_tables(root)
        selection = json.loads((root / "training" / "selection.json").read_text())
        return fit_final_models(cohort, features, selection, config, root / "training")
    if stage == "thresholds":
        return workflow.select_thresholds(root, config)
    if stage == "secondary":
        from .secondary import run_secondary
        return run_secondary(root, config)
    if stage == "explain":
        from .explanations import generate_explanations
        cohort, features = workflow.read_tables(root)
        thresholds = json.loads((root / "development" / "thresholds.json").read_text())
        selection = json.loads((root / "training" / "selection.json").read_text())
        return generate_explanations(workflow.load_models(root), cohort, features, thresholds, config,
                                     root / "explanations", selection["selected_traditional"])
    if stage == "check":
        return workflow.release_checks(root, config)
    if stage == "review":
        return workflow.record_geometry_review(root, args.review_note)
    if stage == "freeze":
        return workflow.freeze_experiment(root, config)
    if stage == "evaluate":
        from .evaluation import evaluate_predictions
        from .explanations import generate_explanations
        frozen = workflow.verify_frozen(root, config)
        cohort, features = workflow.read_tables(root)
        models = workflow.load_models(root)
        thresholds = json.loads((root / "development" / "thresholds.json").read_text())
        predictions = workflow.make_predictions(models, cohort[cohort.partition == "test"], features, config)
        result = evaluate_predictions(predictions, thresholds, config, root / "results", frozen["primary_comparator"])
        generate_explanations(models, cohort, features, thresholds, config,
                              root / "test_explanations", frozen["primary_comparator"], include_test=True)
        return result
    if stage == "benchmark":
        from .benchmark import benchmark_models
        return benchmark_models(root, config)
    if stage == "report":
        from .reporting import write_report
        return write_report(root)
    if stage == "view":
        from .viewer import view_case
        return view_case(root, config, args.nodule_id, args.viewer_output)
    raise ValueError(stage)


def main():
    stages = ["audit", "prepare", "features", "split", "tune", "fit", "thresholds", "secondary",
              "explain", "review", "check", "freeze", "evaluate", "benchmark", "report", "view", "status"]
    parser = argparse.ArgumentParser(description="LIDC radiologist-suspicion research experiment")
    parser.add_argument("stage", choices=stages + ["run"])
    parser.add_argument("--config", default="configs/chapter3.yaml")
    parser.add_argument("--limit-patients", type=int)
    parser.add_argument("--nodule-id")
    parser.add_argument("--viewer-output", default="artifacts/prepared_case.html")
    parser.add_argument("--review-note")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "status":
        from .workflow import experiment_status
        print(json.dumps(experiment_status(config["output_dir"]), indent=2))
        return
    if args.limit_patients is not None and config["profile"] != "smoke":
        raise ValueError("Patient limits require an explicitly separate smoke profile/output")
    root = initialize_run(config)
    if (root / "frozen.json").exists():
        from .workflow import verify_frozen
        verify_frozen(root, config)
        if args.stage not in ("run", "evaluate", "benchmark", "report", "view"):
            raise ValueError("This experiment is frozen; development stages require a separate experiment directory")
    requested = run_stages(root) if args.stage == "run" else [args.stage]
    for stage in requested:
        print(f"Starting {stage}: {root}", flush=True)
        result = execute(stage, root, config, args)
        write_json(root / "status.json", {"last_completed_stage": stage, "profile": config["profile"]})
        write_json(root / "stages" / f"{stage}.json", {"stage": stage, "profile": config["profile"],
                   "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Completed {stage}" + (f": {len(result)} rows" if isinstance(result, pd.DataFrame) else ""), flush=True)
        if args.stage == "run" and stage == "split" and not (root / "data" / "geometry_review.json").exists():
            print("Development inputs are ready. Inspect data/development_overlays.png, record the review using the review stage, then resume run.", flush=True)
            return


if __name__ == "__main__":
    main()
