"""Record the verified Windows environment without machine-specific file URLs."""
from importlib import metadata
from pathlib import Path
import sys


if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Generate the lock with the verified Python 3.11 environment")
if metadata.version("pyradiomics") != "3.1.0":
    raise RuntimeError("Build the pinned PyRadiomics release before freezing dependencies")
packages = {}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if name and name.lower() not in {"lidc-experiment", "pyradiomics", "pip"}:
        packages.setdefault(name.lower(), f"{name}=={distribution.version}")
header = ["# Verified Python 3.11 / Windows x64 environment.",
          "# Install PyRadiomics separately with scripts/build_pyradiomics.ps1.",
          "# The PyPI PyRadiomics source archive has inconsistent version metadata.",
          "--extra-index-url https://download.pytorch.org/whl/cu128", ""]
target = Path(__file__).resolve().parents[1] / "requirements-windows.lock.txt"
target.write_text("\n".join(header + [packages[key] for key in sorted(packages)]) + "\n", encoding="utf-8")
print(target)
