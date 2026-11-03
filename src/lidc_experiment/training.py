"""Patient-grouped model development for the prespecified Chapter 3 experiment.

The public entry points accept dataframes rather than notebook state.  Every fitted
feature transformation and calibration fold carries its training identifiers.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.svm import SVC

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .resources import ResourceMonitor


MODEL_ORDER = ("logistic_regression", "svm", "random_forest", "cnn")
MODEL_ALIASES = {"lr": "logistic_regression", "logreg": "logistic_regression",
                 "rf": "random_forest", "randomforest": "random_forest"}


def _canonical(name: str) -> str:
    name = MODEL_ALIASES.get(name, name)
    if name not in MODEL_ORDER:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_ORDER}.")
    return name


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _settings(config: dict) -> tuple[dict, dict, int]:
    training = config.get("training", {})
    cnn = {"batch_size": 32, "max_epochs": 60, "patience": 8,
           "min_delta": 0.0001, "weight_decay": 0.0001}
    cnn.update(config.get("cnn", {}))
    cnn.update(training.get("cnn", {}))
    runtime = {"device": "cuda" if torch.cuda.is_available() else "cpu",
               "num_workers": 0, "cpu_threads": 1}
    runtime.update(config.get("runtime", {}))
    seed = int(training.get("seed", config.get("seed", 42)))
    if str(config.get("profile", "full")) != "smoke":
        required = {"batch_size": 32, "max_epochs": 60, "patience": 8,
                    "min_delta": 0.0001, "weight_decay": 0.0001}
        if any(cnn[key] != value for key, value in required.items()):
            raise ValueError("Reduced or changed CNN training limits require profile='smoke'.")
    return cnn, runtime, seed


def _seed_everything(seed: int, runtime: dict) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(runtime["cpu_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(str(runtime["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no automatic CPU substitution.")
    return device


def _validate_cohort(cohort: pd.DataFrame, *, training_only: bool = False) -> None:
    required = {"patient_id", "nodule_id", "label"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"Cohort is missing columns: {sorted(required - set(cohort.columns))}")
    if cohort.empty or cohort["nodule_id"].duplicated().any():
        raise ValueError("Cohort must be nonempty with unique nodule identifiers.")
    if cohort[list(required)].isna().any().any():
        raise ValueError("Patient, nodule and label fields may not be missing.")
    if not set(cohort["label"].unique()).issubset({0, 1}):
        raise ValueError("Only binary low/high-suspicion labels 0 and 1 are permitted.")
    if "partition" in cohort:
        if not set(cohort["partition"].unique()).issubset({"train", "validation", "test"}):
            raise ValueError("Unknown partition; use train, validation or test.")
        for key in ("patient_id", "series_uid"):
            if key in cohort and (cohort.groupby(key)["partition"].nunique() > 1).any():
                raise ValueError(f"Leakage: {key} occurs in multiple partitions.")
        if training_only and set(cohort["partition"]) != {"train"}:
            raise ValueError("Model fitting accepts training observations only.")


def _feature_frame(cohort: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None or "nodule_id" not in features:
        raise ValueError("A nodule-keyed radiomics feature table is required.")
    if features["nodule_id"].duplicated().any():
        raise ValueError("Feature table contains duplicated nodule identifiers.")
    columns = sorted(c for c in features if c.startswith("original_"))
    if not columns:
        raise ValueError("No allowlisted original_* radiomics features were supplied.")
    indexed = features.set_index("nodule_id")
    missing = set(cohort["nodule_id"]) - set(indexed.index)
    if missing:
        raise ValueError(f"Missing radiomics rows for {len(missing)} cohort nodules.")
    result = indexed.loc[cohort["nodule_id"].tolist(), columns].copy()
    return result.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)


class RadiomicsPreprocessor(BaseEstimator, TransformerMixin):
    """Training-only imputation, deterministic correlation pruning and scaling."""

    def __init__(self, scale: bool = True, correlation_threshold: float = 0.90):
        self.scale = scale
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y=None, *, patient_ids=None, nodule_ids=None):
        frame = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        self.input_features_ = sorted(frame.columns.tolist())
        if any(not str(c).startswith("original_") for c in self.input_features_):
            raise ValueError("Only explicit original_* radiomics predictors may enter a fit.")
        usable = [c for c in self.input_features_ if frame[c].dropna().nunique() > 1]
        if not usable:
            raise ValueError("No nonconstant finite training features remain.")
        self.medians_ = frame[usable].median()
        imputed = frame[usable].fillna(self.medians_)
        correlation = imputed.corr().abs()
        retained: list[str] = []
        for column in usable:
            if not any(correlation.loc[column, old] > self.correlation_threshold for old in retained):
                retained.append(column)
        self.retained_features_ = retained
        self.feature_names_out_ = np.asarray(retained, dtype=object)
        values = imputed[retained].to_numpy(dtype=np.float64)
        self.means_ = values.mean(axis=0)
        self.scales_ = values.std(axis=0)
        self.scales_[self.scales_ == 0] = 1.0
        self.fit_patient_ids_ = sorted(set(map(str, patient_ids))) if patient_ids is not None else []
        self.fit_nodule_ids_ = list(map(str, nodule_ids)) if nodule_ids is not None else list(map(str, frame.index))
        self.n_features_in_ = len(self.input_features_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
        absent = set(self.retained_features_) - set(frame.columns)
        if absent:
            raise ValueError(f"Prediction table lacks fitted features: {sorted(absent)}")
        values = frame[self.retained_features_].fillna(self.medians_).to_numpy(dtype=np.float64)
        if self.scale:
            values = (values - self.means_) / self.scales_
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite feature values remain after fitted preprocessing.")
        return values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_.copy()


class SigmoidCalibrator:
    """Platt sigmoid fit to grouped out-of-fold scores, using smoothed targets."""

    def fit(self, scores, labels):
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.int64)
        negatives, positives = int((labels == 0).sum()), int((labels == 1).sum())
        if not negatives or not positives or not np.isfinite(scores).all():
            raise ValueError("Sigmoid calibration requires finite OOF scores from both classes.")
        self.score_scale_ = max(1.0, float(np.max(np.abs(scores))))
        x = scores / self.score_scale_
        targets = np.where(labels == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))

        def objective(parameters):
            logits = parameters[0] * x + parameters[1]
            loss = np.logaddexp(0, logits).sum() - np.dot(targets, logits)
            error = expit(logits) - targets
            return float(loss), np.array([np.dot(error, x), error.sum()])

        result = minimize(objective, np.array([0.0, np.log((positives + 1) / (negatives + 1))]),
                          method="BFGS", jac=True, options={"gtol": 1e-6, "maxiter": 1000})
        if not result.success:
            raise RuntimeError(f"Sigmoid calibration failed: {result.message}")
        self.coef_, self.intercept_ = map(float, result.x)
        self.optimization_ = {"method": "BFGS", "gradient_tolerance": 1e-6,
                              "gradient_inf_norm": float(np.linalg.norm(result.jac, ord=np.inf)),
                              "iterations": int(result.nit), "objective": float(result.fun), "converged": True}
        return self

    def predict(self, scores):
        return expit(self.coef_ * np.asarray(scores, dtype=np.float64) / self.score_scale_ + self.intercept_)


class CompactCNN(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, image):
        return self.head(self.pool(self.blocks(image))).squeeze(1)


class NoduleDataset(Dataset):
    def __init__(self, cohort: pd.DataFrame, augment: bool = False):
        self.augment = augment
        self.images = []
        self.labels = cohort["label"].to_numpy(dtype=np.float32)
        for filename in cohort["npz_path"]:
            with np.load(filename, allow_pickle=False) as case:
                image = np.asarray(case["image"], dtype=np.float32)
            if image.shape != (2, 64, 64) or not np.isfinite(image).all():
                raise ValueError(f"Invalid CNN input {filename}: expected finite (2,64,64).")
            if not np.isin(image[1], [0, 1]).all() or not np.any(image[1]):
                raise ValueError(f"CNN input has an invalid or empty mask: {filename}")
            if np.any(image[0] < 0) or np.any(image[0] > 1) or np.any(image[0][image[1] == 0] != 0):
                raise ValueError(f"CNN intensity channel violates the masked [0,1] representation: {filename}")
            self.images.append(torch.from_numpy(image.copy()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), (1, 2))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, (2,))
            if torch.rand(()) < 0.5:
                image = torch.flip(image, (1,))
        return image.contiguous(), torch.tensor(self.labels[index])


@dataclass
class ModelBundle:
    model_name: str
    estimator: Any = None
    feature_processor: RadiomicsPreprocessor | None = None
    network: CompactCNN | None = None
    calibrator: SigmoidCalibrator | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def cnnmodule(self):
        return self.network

    @property
    def retained_features(self):
        return self.feature_processor.retained_features_ if self.feature_processor is not None else []


def _grids(config: dict) -> dict[str, list[dict]]:
    grids = {
        "logistic_regression": [dict(C=c, class_weight=w) for c, w in itertools.product(
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], [None, "balanced"])],
        "svm": [dict(C=c, gamma=g, class_weight=w) for c, g, w in itertools.product(
            [0.1, 1.0, 10.0], ["scale", 0.01], [None, "balanced"])],
        "random_forest": [dict(max_depth=d, min_samples_leaf=l, class_weight=w)
                          for d, l, w in itertools.product([None, 5, 10], [1, 5], [None, "balanced"])],
        "cnn": [dict(learning_rate=r, dropout=d, weighted_loss=w) for r, d, w in itertools.product(
            [0.0001, 0.0003, 0.001], [0.2, 0.5], [False, True])],
    }
    overrides = config.get("grids", config.get("training", {}).get("grids", {}))
    if overrides and config.get("profile", "full") != "smoke":
        raise ValueError("Candidate-grid overrides require profile='smoke'.")
    for model_name, settings in overrides.items():
        key = _canonical(model_name)
        if isinstance(settings, dict):
            keys = list(settings)
            settings = [dict(zip(keys, values)) for values in itertools.product(*(settings[k] for k in keys))]
        if not settings:
            raise ValueError(f"Empty candidate grid for {key}.")
        grids[key] = copy.deepcopy(settings)
    return grids


def _training_folds(train: pd.DataFrame, seed: int, *, use_stored: bool = True):
    if use_stored and "fold" in train:
        if train["fold"].isna().any() or not set(train["fold"]).issubset({0, 1, 2}):
            raise ValueError("Training observations require persisted fold values 0, 1 or 2.")
        if (train.groupby("patient_id")["fold"].nunique() > 1).any():
            raise ValueError("Leakage: a patient occurs in multiple training folds.")
        if set(train["fold"]) != {0, 1, 2}:
            raise ValueError("All three persisted training folds must be represented.")
        splits = [(np.flatnonzero(train["fold"].to_numpy() != fold),
                   np.flatnonzero(train["fold"].to_numpy() == fold)) for fold in range(3)]
    else:
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = list(splitter.split(train, train["label"], train["patient_id"]))
    for fit, held in splits:
        if set(train.iloc[fit]["patient_id"]) & set(train.iloc[held]["patient_id"]):
            raise ValueError("Leakage in patient-grouped cross-validation.")
        if train.iloc[fit]["label"].nunique() != 2 or train.iloc[held]["label"].nunique() != 2:
            raise ValueError("Every training and held-out fold must contain both classes.")
    return splits


def _inner_patients(train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = train.groupby("patient_id", sort=True)["label"].max()
    if patients.value_counts().min() < 2:
        raise ValueError("Insufficient patient strata for the CNN inner 10% split.")
    try:
        fit_ids, held_ids = train_test_split(patients.index.to_numpy(), test_size=0.10,
                                           random_state=seed, stratify=patients.to_numpy())
    except ValueError as exc:
        raise ValueError("CNN inner 10% grouped stratification is infeasible; revise before tuning.") from exc
    fit = train[train["patient_id"].isin(fit_ids)].copy()
    held = train[train["patient_id"].isin(held_ids)].copy()
    if fit["label"].nunique() != 2 or held["label"].nunique() != 2:
        raise ValueError("CNN inner training/validation must both contain both nodule classes.")
    return fit, held


def _cnn_predict(network: CompactCNN, cohort: pd.DataFrame, config: dict) -> np.ndarray:
    _, runtime, _ = _settings(config)
    device = torch.device(runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for CNN prediction but unavailable.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    network.to(device).eval()
    loader = DataLoader(NoduleDataset(cohort), batch_size=int(config.get("inference_batch_size", 32)),
                        shuffle=False, num_workers=int(runtime["num_workers"]))
    values = []
    with torch.inference_mode():
        for images, _ in loader:
            values.append(network(images.to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def _train_cnn(train: pd.DataFrame, settings: dict, config: dict, seed: int,
               *, augment: bool, epochs: int, early_validation: pd.DataFrame | None = None):
    cnn, runtime, _ = _settings(config)
    device = _seed_everything(seed, runtime)
    network = CompactCNN(float(settings["dropout"])).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(NoduleDataset(train, augment=augment), batch_size=int(cnn["batch_size"]),
                        shuffle=True, num_workers=int(runtime["num_workers"]), generator=generator)
    positives = int(train["label"].sum())
    negatives = len(train) - positives
    if not positives or not negatives:
        raise ValueError("CNN fitting requires both classes.")
    positive_weight = negatives / positives if settings.get("weighted_loss", False) else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(network.parameters(), lr=float(settings["learning_rate"]),
                                 weight_decay=float(cnn["weight_decay"]))
    history, best_epoch, best_auc, stale = [], 0, -np.inf, 0
    for epoch in range(1, int(epochs) + 1):
        network.train()
        losses = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite CNN training loss.")
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach().cpu()), len(labels)))
        row = {"epoch": epoch, "loss": sum(v * n for v, n in losses) / len(train)}
        if early_validation is not None:
            auc = float(roc_auc_score(early_validation["label"], _cnn_predict(network, early_validation, config)))
            row["inner_validation_auc"] = auc
            if best_epoch == 0 or auc >= best_auc + float(cnn["min_delta"]):
                best_epoch, best_auc, stale = epoch, auc, 0
            else:
                stale += 1
        history.append(row)
        if early_validation is not None and stale >= int(cnn["patience"]):
            break
    if early_validation is None:
        best_epoch = int(epochs)
    metadata = {"history": history, "best_epoch": best_epoch,
                "positive_class_weight": float(positive_weight), "epochs_run": len(history),
                "device": str(device), "deterministic_algorithms": True,
                "cudnn_deterministic": True, "cudnn_benchmark": False}
    if early_validation is not None:
        metadata.update(best_inner_auc=float(best_auc),
                        early_stopping_patient_ids=sorted(map(str, early_validation["patient_id"].unique())))
    return network.cpu(), metadata


def _new_estimator(model_name: str, settings: dict, seed: int, runtime: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(C=float(settings["C"]), class_weight=settings.get("class_weight"),
                                  solver="lbfgs", max_iter=5000, random_state=seed)
    if model_name == "svm":
        return SVC(C=float(settings["C"]), gamma=settings["gamma"], kernel="rbf",
                   class_weight=settings.get("class_weight"), probability=False, random_state=seed)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=500, max_depth=settings.get("max_depth"),
                                      min_samples_leaf=int(settings["min_samples_leaf"]),
                                      class_weight=settings.get("class_weight"), max_features="sqrt",
                                      random_state=seed, n_jobs=int(runtime["cpu_threads"]))
    raise ValueError(f"No traditional estimator for {model_name}.")


def _fit_ml(model_name, train, features, settings, config, seed):
    _, runtime, _ = _settings(config)
    raw = _feature_frame(train, features)
    processor = RadiomicsPreprocessor(scale=model_name != "random_forest").fit(
        raw, patient_ids=train["patient_id"], nodule_ids=train["nodule_id"])
    estimator = _new_estimator(model_name, settings, seed, runtime)
    estimator.fit(processor.transform(raw), train["label"].to_numpy(dtype=int))
    return estimator, processor


def _fit_fingerprint(model_name, train, features, settings, config, seed, augment, epochs, calibrate):
    key_columns = [c for c in ("nodule_id", "patient_id", "label", "fold", "npz_path") if c in train]
    fingerprint = {"model": model_name, "cohort": train[key_columns].astype(str).to_dict("records"),
                   "settings": settings, "config": config, "seed": seed, "augment": augment,
                   "epochs": epochs, "calibrate": calibrate,
                   "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "versions": {"numpy": np.__version__, "torch": torch.__version__}}
    if model_name != "cnn":
        raw = _feature_frame(train, features)
        fingerprint["features_hash"] = hashlib.sha256(
            pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest()
        fingerprint["feature_columns"] = raw.columns.tolist()
    else:
        fingerprint["input_hashes"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in train["npz_path"]]
    return _hash(fingerprint)


def _save_model(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if bundle.model_name == "cnn":
        torch.save({"format": "lidc_cnn_v1", "model_name": "cnn",
                    "dropout": float(bundle.metadata["settings"]["dropout"]),
                    "state_dict": bundle.network.cpu().state_dict(), "metadata": _jsonable(bundle.metadata)}, tmp)
    else:
        joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)
    _write_json(path.with_suffix(path.suffix + ".metadata.json"), bundle.metadata)


def load_model(artifact_path: str | Path) -> ModelBundle:
    path = Path(artifact_path)
    if path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != "lidc_cnn_v1":
            raise ValueError("Unsupported CNN checkpoint format.")
        network = CompactCNN(float(saved["dropout"]))
        network.load_state_dict(saved["state_dict"])
        network.eval()
        return ModelBundle("cnn", network=network, metadata=saved["metadata"])
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("Artifact is not a LIDC ModelBundle.")
    return bundle


def fit_selected_model(model_name: str, training_cohort: pd.DataFrame,
                       training_features: pd.DataFrame | None, settings: dict,
                       config: dict, output_path: str | Path | None,
                       seed: int = 42, augment: bool = True, epochs: int | None = None,
                       *, calibrate: bool = True) -> ModelBundle:
    """Fit selected settings on training patients, optionally persisting a model.

    Secondary analyses call this same function so their transformations and class
    weights are refitted on precisely their own training observations.
    """
    model_name = _canonical(model_name)
    train = training_cohort.reset_index(drop=True).copy()
    _validate_cohort(train, training_only=True)
    if train["label"].nunique() != 2:
        raise ValueError("Fitting requires both classes.")
    _settings(config)
    if model_name == "cnn" and (epochs is None or int(epochs) < 1):
        raise ValueError("Final CNN fitting requires an explicitly selected positive epoch count.")
    fingerprint = _fit_fingerprint(model_name, train, training_features, settings, config,
                                   seed, augment, epochs, calibrate)
    path = Path(output_path) if output_path is not None else None
    if path is not None and model_name == "cnn" and path.suffix != ".pt":
        raise ValueError("CNN checkpoint paths must end in .pt.")
    if path is not None and path.exists():
        saved = load_model(path)
        if saved.metadata.get("fit_fingerprint") != fingerprint:
            raise ValueError(f"Refusing incompatible artifact reuse: {path}")
        return saved
    metadata = {"model_name": model_name, "settings": _jsonable(settings), "seed": int(seed),
                "fit_fingerprint": fingerprint, "fit_patient_ids": sorted(map(str, train["patient_id"].unique())),
                "fit_nodule_ids": list(map(str, train["nodule_id"])), "training_nodules": len(train),
                "training_positives": int(train["label"].sum()), "profile": config.get("profile", "full"),
                "numerical_precision": "float32" if model_name == "cnn" else "float64",
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, ResourceMonitor(
            cuda=model_name == "cnn" and str(_settings(config)[1]["device"]).startswith("cuda")) as resources:
        warnings.simplefilter("always")
        if model_name == "cnn":
            network, details = _train_cnn(train, settings, config, seed, augment=augment, epochs=int(epochs))
            metadata.update(details, augment=bool(augment), final_epochs=int(epochs))
            bundle = ModelBundle(model_name, network=network, metadata=metadata)
        else:
            estimator, processor = _fit_ml(model_name, train, training_features, settings, config, seed)
            bundle = ModelBundle(model_name, estimator=estimator, feature_processor=processor, metadata=metadata)
            metadata["retained_features"] = processor.retained_features_
            if model_name == "svm" and calibrate:
                oof = np.full(len(train), np.nan, dtype=float)
                calibration_folds = []
                for fold, (fit, held) in enumerate(_training_folds(train, seed)):
                    subset = train.iloc[fit]
                    calibration_estimator, calibration_processor = _fit_ml(
                        model_name, subset, training_features, settings, config, seed)
                    oof[held] = calibration_estimator.decision_function(
                        calibration_processor.transform(_feature_frame(train.iloc[held], training_features)))
                    calibration_folds.append({"fold": fold,
                                              "fit_patient_ids": sorted(map(str, subset["patient_id"].unique())),
                                              "held_out_patient_ids": sorted(map(str, train.iloc[held]["patient_id"].unique())),
                                              "retained_features": calibration_processor.retained_features_})
                if not np.isfinite(oof).all():
                    raise RuntimeError("Grouped SVM calibration did not produce every OOF score.")
                bundle.calibrator = SigmoidCalibrator().fit(oof, train["label"])
                metadata["calibration_folds"] = calibration_folds
                metadata["calibration_method"] = "grouped_oof_platt_sigmoid"
                metadata["calibration_optimization"] = bundle.calibrator.optimization_
                metadata["calibration_oof"] = [{"nodule_id": str(n), "score": float(s), "label": int(y)}
                                                for n, s, y in zip(train["nodule_id"], oof, train["label"])]
        metadata["warnings"] = sorted(set(str(w.message) for w in caught))
    metadata["fit_wall_seconds"] = time.perf_counter() - start
    metadata.update(resources.as_dict())
    if path is not None:
        _save_model(bundle, path)
    return bundle


def predict_model(model: ModelBundle, cohort: pd.DataFrame, features: pd.DataFrame | None,
                  config: dict) -> pd.DataFrame:
    """Return native ranking scores and high-suspicion probabilities in input order."""
    _validate_cohort(cohort)
    if model.model_name == "cnn":
        logits = _cnn_predict(model.network, cohort.reset_index(drop=True), config)
        probability = expit(logits)
        score = probability.copy()
    else:
        values = model.feature_processor.transform(_feature_frame(cohort, features))
        if model.model_name == "svm":
            score = np.asarray(model.estimator.decision_function(values), dtype=float)
            probability = (model.calibrator.predict(score) if model.calibrator is not None
                           else np.full(len(score), np.nan))
        else:
            positive_column = int(np.flatnonzero(model.estimator.classes_ == 1)[0])
            probability = np.asarray(model.estimator.predict_proba(values)[:, positive_column], dtype=float)
            score = probability.copy()
    if not np.isfinite(score).all():
        raise RuntimeError("Model produced nonfinite native scores.")
    if model.model_name != "svm" or model.calibrator is not None:
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise RuntimeError("Model produced invalid probabilities.")
    return pd.DataFrame({"nodule_id": cohort["nodule_id"].to_numpy(), "score": score,
                         "probability": probability})


def tune_models(cohort: pd.DataFrame, features: pd.DataFrame | None, config: dict,
                output_dir: str | Path, models=None) -> dict:
    """Evaluate the prespecified candidate grids on the same stored patient folds."""
    _validate_cohort(cohort)
    if "partition" not in cohort:
        raise ValueError("Tuning requires an explicit patient partition manifest.")
    train = cohort[cohort["partition"] == "train"].reset_index(drop=True).copy()
    _validate_cohort(train, training_only=True)
    cnn, _, seed = _settings(config)
    grids = _grids(config)
    chosen_models = list(MODEL_ORDER if models is None else dict.fromkeys(_canonical(m) for m in models))
    if not chosen_models:
        raise ValueError("Select at least one model to tune.")
    splits = _training_folds(train, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, selections, search_seconds, invocation_seconds, fold_seconds = [], {}, {}, {}, {}
    for model_name in chosen_models:
        model_start = time.perf_counter()
        reused_rows = 0
        candidates = []
        for index, settings in enumerate(grids[model_name]):
            candidate_rows = []
            for fold, (fit_indices, held_indices) in enumerate(splits):
                fit, held = train.iloc[fit_indices].copy(), train.iloc[held_indices].copy()
                folder = output / "cv_fits" / model_name / f"candidate_{index:02d}" / f"fold_{fold}"
                record_path = folder / "record.json"
                fingerprint = _fit_fingerprint(model_name, train, features, settings, config,
                                               seed, True, None, False)
                fingerprint = _hash({"candidate": fingerprint, "fold": fold})
                if record_path.exists():
                    row = json.loads(record_path.read_text(encoding="utf-8"))
                    if row.get("fingerprint") != fingerprint:
                        raise ValueError(f"Refusing incompatible CV resume at {record_path}")
                    if row["status"] != "success":
                        record_path.unlink()
                    else:
                        rows.append(row)
                        candidate_rows.append(row)
                        reused_rows += 1
                        continue
                folder.mkdir(parents=True, exist_ok=True)
                row = {"model": model_name, "candidate": index, "fold": fold, "settings": settings,
                       "fingerprint": fingerprint, "status": "failed", "auc": None, "best_epoch": None}
                started = time.perf_counter()
                print(f"[{model_name}] candidate {index + 1}/{len(grids[model_name])}, fold {fold + 1}/3", flush=True)
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        suffix = ".pt" if model_name == "cnn" else ".joblib"
                        model_path = folder / ("model" + suffix)
                        if model_name == "cnn":
                            inner_fit, inner_validation = _inner_patients(fit, seed)
                            if set(inner_validation["patient_id"]) & set(held["patient_id"]):
                                raise ValueError("Outer held-out patients entered CNN early stopping.")
                            _, early_details = _train_cnn(inner_fit, settings, config, seed, augment=True,
                                                          epochs=int(cnn["max_epochs"]), early_validation=inner_validation)
                            row["best_epoch"] = int(early_details["best_epoch"])
                            _write_json(folder / "early_stopping.json", {
                                **early_details, "fit_patient_ids": sorted(map(str, inner_fit["patient_id"].unique())),
                                "outer_held_out_patient_ids": sorted(map(str, held["patient_id"].unique()))})
                            model = fit_selected_model(model_name, fit, features, settings, config, model_path,
                                                       seed=seed, epochs=row["best_epoch"], calibrate=False)
                        else:
                            model = fit_selected_model(model_name, fit, features, settings, config, model_path,
                                                       seed=seed, calibrate=False)
                        predictions = predict_model(model, held, features, config)
                        predictions["patient_id"] = held["patient_id"].to_numpy()
                        predictions["label"] = held["label"].to_numpy()
                        predictions.to_csv(folder / "predictions.csv", index=False)
                        row.update(status="success", auc=float(roc_auc_score(held["label"], predictions["score"])),
                                   warnings=sorted(set(str(w.message) for w in caught)))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["wall_seconds"] = time.perf_counter() - started
                _write_json(record_path, row)
                rows.append(row)
                candidate_rows.append(row)
                pd.DataFrame([{**r, "settings": json.dumps(r["settings"], sort_keys=True)} for r in rows]).to_csv(
                    output / "cv_results.csv", index=False)
            if len(candidate_rows) == 3 and all(r["status"] == "success" for r in candidate_rows):
                candidates.append({"candidate_index": index, "settings": settings,
                                   "mean_auc": float(np.mean([r["auc"] for r in candidate_rows])),
                                   "fold_aucs": [r["auc"] for r in candidate_rows],
                                   "fold_best_epochs": [r["best_epoch"] for r in candidate_rows]})
        invocation_seconds[model_name] = time.perf_counter() - model_start
        fold_seconds[model_name] = float(sum(r["wall_seconds"] for r in rows if r["model"] == model_name))
        timing_path = output / "search_timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        previous = timing.get(model_name, {})
        if not reused_rows:
            timing[model_name] = {"total_search_wall_seconds": invocation_seconds[model_name], "timing_complete": True}
        elif previous:
            # A fully resumed comparison retains its original measured cost.
            if reused_rows < len(grids[model_name]) * 3 and timing[model_name]["total_search_wall_seconds"] is not None:
                timing[model_name]["total_search_wall_seconds"] += invocation_seconds[model_name]
                timing[model_name]["timing_complete"] = False
        else:
            timing[model_name] = {"total_search_wall_seconds": None, "timing_complete": False}
        _write_json(timing_path, timing)
        search_seconds[model_name] = timing[model_name]["total_search_wall_seconds"]
        if not candidates:
            _write_json(output / "search_status.json", {"failed_model": model_name, "search_seconds": search_seconds})
            raise RuntimeError(f"All {model_name} candidates failed; inspect {output / 'cv_results.csv'}.")
        highest = max(c["mean_auc"] for c in candidates)
        best = next(c for c in candidates if highest - c["mean_auc"] <= 0.0001)
        if model_name == "cnn":
            best["epochs"] = max(1, int(np.median(best["fold_best_epochs"])))
        selections[model_name] = best
    traditional = [name for name in MODEL_ORDER[:3] if name in selections]
    comparator = None
    if traditional:
        highest = max(selections[name]["mean_auc"] for name in traditional)
        comparator = next(name for name in traditional if highest - selections[name]["mean_auc"] <= 0.0001)
    result = {"profile": config.get("profile", "full"), "seed": seed, "models": selections,
              "selected_traditional": comparator, "search_wall_seconds": search_seconds,
              "cv_fit_and_scoring_seconds": fold_seconds,
              "invocation_wall_seconds": invocation_seconds,
              "training_patient_ids": sorted(map(str, train["patient_id"].unique())),
              "selection_metric": "mean_training_grouped_cv_roc_auc", "tie_tolerance": 0.0001}
    _write_json(output / "selection.json", result)
    return result


def fit_final_models(cohort: pd.DataFrame, features: pd.DataFrame | None, selection: dict,
                     config: dict, output_dir: str | Path) -> dict:
    """Refit each selected pipeline on training patients without validation/test access."""
    _validate_cohort(cohort)
    if "partition" not in cohort:
        raise ValueError("Final fitting requires the persisted partition manifest.")
    train = cohort[cohort["partition"] == "train"].reset_index(drop=True).copy()
    _, _, seed = _settings(config)
    output = Path(output_dir)
    paths = {}
    for name in MODEL_ORDER:
        if name not in selection["models"]:
            continue
        selected = selection["models"][name]
        path = output / "models" / (name + (".pt" if name == "cnn" else ".joblib"))
        fit_selected_model(name, train, features, selected["settings"], config, path,
                           seed=seed, epochs=selected.get("epochs"), calibrate=True)
        paths[name] = str(path.resolve())
    _write_json(output / "model_artifacts.json", paths)
    return paths
