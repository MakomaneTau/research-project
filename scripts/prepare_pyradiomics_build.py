"""Correct release packaging metadata; leave scientific source files untouched."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".build" / "pyradiomics"
REVISION = "6a761c4eb01905cfcb1139b0b158547c55edd4df"


def main():
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError("Unexpected source revision")
    fixes = {"pyproject.toml": [('version = "3.0.1a1"', 'version = "3.1.0"')],
             "setup.cfg": [("tag_prefix = ''", "tag_prefix = v")]}
    for filename, replacements in fixes.items():
        path = SOURCE / filename
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            if old not in text and new not in text:
                raise RuntimeError(f"Unexpected upstream packaging text: {filename}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    changed = subprocess.check_output(["git", "-C", str(SOURCE), "diff", "--name-only"], text=True).splitlines()
    if set(changed) - set(fixes):
        raise RuntimeError("Scientific source files differ from the pinned release")
    patch = subprocess.check_output(["git", "-C", str(SOURCE), "diff", "--", *fixes], text=True)
    record = {"upstream_commit": REVISION, "release_tag": "v3.1.0", "scientific_source_modified": False,
              "packaging_changes": {"pyproject.toml": "Correct static project version to released 3.1.0",
                                    "setup.cfg": "Recognize the upstream v-prefixed release tag"},
              "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "patch": patch}
    (ROOT / ".build" / "pyradiomics-build-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Verified release source and recorded packaging-only corrections")


if __name__ == "__main__":
    main()
