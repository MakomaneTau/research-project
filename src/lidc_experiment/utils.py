"""Configuration, atomic artifacts and reproducibility records."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

MODELS = ("logistic_regression", "svm", "random_forest", "cnn")


def atomic_replace(source, destination):
    """Tolerate short Windows reader/antivirus locks without losing a snapshot."""
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2 ** attempt)


def read_csv_snapshot(path):
    import pandas as pd
    for attempt in range(6):
        try:
            return pd.read_csv(path)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2 ** attempt)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=json_default, allow_nan=False), encoding="utf-8")
    atomic_replace(temporary, path)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=json_default).encode()).hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("profile") not in ("primary", "smoke"):
        raise ValueError("profile must be primary or smoke")
    if config.get("profile") == "primary":
        p = config["preprocessing"]
        required = {"spacing_mm": 1, "hu_min": -1000, "hu_max": 400,
                    "crop_size": 64, "max_slice_spacing_mm": 2.5, "min_readers": 2, "bin_width": 25}
        if any(p.get(key) != value for key, value in required.items()):
            raise ValueError("Primary preprocessing differs from the declared Chapter 3 protocol")
        evaluation = {"target_sensitivity": .9, "bootstrap_replicates": 2000,
                      "bootstrap_max_attempts": 20000, "bootstrap_seed": 42}
        if any(config.get("evaluation", {}).get(key, value) != value for key, value in evaluation.items()):
            raise ValueError("Primary evaluation differs from the declared Chapter 3 protocol")
        if config.get("training", {}).get("seed", 42) != 42:
            raise ValueError("Primary seed must remain 42")
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    return config


def environment_record():
    packages = {}
    seen = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and name.lower() not in seen:
            packages[name] = distribution.version
            seen.add(name.lower())
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "executable": sys.executable, "platform": platform.platform(),
              "processor": platform.processor(), "cpu_count": os.cpu_count(),
              "packages": dict(sorted(packages.items()))}
    build_record = Path(__file__).resolve().parents[2] / ".build" / "pyradiomics-build-source.json"
    if build_record.exists():
        result["pyradiomics_build"] = json.loads(build_record.read_text())
    wheel_record = build_record.with_name("pyradiomics-wheel-sha256.json")
    if wheel_record.exists():
        result["pyradiomics_wheel"] = json.loads(wheel_record.read_text(encoding="utf-8-sig"))
    try:
        import psutil
        result["ram_bytes"] = psutil.virtual_memory().total
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_numerics"] = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                    "automatic_mixed_precision": False}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = {"name": props.name, "memory_bytes": props.total_memory}
    except ImportError:
        pass
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def initialize_run(config):
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_hash(config)
    marker = root / "run.json"
    if marker.exists() and json.loads(marker.read_text())["config_hash"] != fingerprint:
        raise ValueError("Output directory contains another configuration; use a new output directory")
    if not marker.exists():
        write_json(marker, {"config_hash": fingerprint, "config": config,
                            "created_utc": datetime.now(timezone.utc).isoformat()})
        write_json(root / "environment.json", environment_record())
        (root / "requirements.lock.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    if not (root / "frozen.json").exists():
        write_json(root / "environments" / f"python_{sys.version_info.major}_{sys.version_info.minor}.json", environment_record())
    return root
