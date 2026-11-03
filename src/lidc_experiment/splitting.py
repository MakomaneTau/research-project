"""One immutable patient allocation shared by every model and analysis."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def assert_independent(cohort):
    if cohort["nodule_id"].duplicated().any():
        raise ValueError("Duplicate nodule identities")
    if cohort.groupby("patient_id")["partition"].nunique().gt(1).any():
        raise ValueError("Patient overlap across partitions")
    if "series_uid" in cohort and cohort.groupby("series_uid")["partition"].nunique().gt(1).any():
        raise ValueError("Source series overlap across partitions")
    if "source_image_uids" in cohort:
        sources = {}
        for row in cohort.itertuples():
            identifiers = json.loads(row.source_image_uids) if isinstance(row.source_image_uids, str) else row.source_image_uids
            for identifier in identifiers:
                previous = sources.setdefault(identifier, row.partition)
                if previous != row.partition:
                    raise ValueError("Source image overlap across partitions")
    for partition in ("train", "validation", "test"):
        rows = cohort[cohort.partition == partition]
        if set(rows.label.unique()) != {0, 1}:
            raise ValueError(f"{partition} must contain both classes")
    training = cohort[cohort.partition == "train"]
    if training.groupby("patient_id")["fold"].nunique().gt(1).any():
        raise ValueError("Patient overlap across training folds")
    for fold in range(3):
        for rows in (training[training.fold == fold], training[training.fold != fold]):
            if set(rows.label.unique()) != {0, 1}:
                raise ValueError(f"Training fold {fold} must contain both classes")


def assign_partitions(cohort, seed=42):
    result = cohort.copy().sort_values("nodule_id").reset_index(drop=True)
    if not result.label.isin([0, 1]).all():
        raise ValueError("Partition the final binary cohort only")
    patients = result.groupby("patient_id", sort=True).label.max()
    train, held = train_test_split(patients.index.to_numpy(), test_size=0.30, random_state=seed,
                                  stratify=patients.to_numpy())
    validation, test = train_test_split(held, test_size=0.5, random_state=seed,
                                       stratify=patients.loc[held].to_numpy())
    mapping = {p: name for name, ids in (("train", train), ("validation", validation), ("test", test)) for p in ids}
    result["partition"] = result.patient_id.map(mapping)
    result["fold"] = -1
    training = result[result.partition == "train"]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(training)), training.label, training.patient_id)):
        result.loc[training.index[valid], "fold"] = fold
    assert_independent(result)
    return result


def allocation_summary(cohort):
    return {name: {"patients": int(rows.patient_id.nunique()), "nodules": len(rows),
                   "low": int((rows.label == 0).sum()), "high": int((rows.label == 1).sum())}
            for name, rows in cohort.groupby("partition")}


def nested_patient_subsets(training, fractions=(0.25, 0.5, 0.75, 1.0), seed=42):
    patients = training.groupby("patient_id", sort=True).label.max()
    rng = np.random.default_rng(seed)
    strata = [rng.permutation(patients[patients == value].index.to_numpy()) for value in (0, 1)]
    subsets = {}
    for fraction in fractions:
        ids = np.concatenate([s[:max(1, int(np.ceil(float(fraction) * len(s))))] for s in strata])
        selected = training[training.patient_id.isin(ids)].copy()
        if set(selected.label.unique()) != {0, 1}:
            raise ValueError("Learning-curve subset does not contain both classes")
        subsets[float(fraction)] = selected
    return subsets
