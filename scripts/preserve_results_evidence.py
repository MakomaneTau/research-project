"""Read-only experiment preservation. Run once before evidence recalculation."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/chapter3'
OUT = ROOT / 'results_evidence/2026-11-12'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f'Original experiment directory is unavailable: {SOURCE}')
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'preservation'
    target.mkdir(exist_ok=True)
    archive = target / 'chapter3_original.zip'
    manifest = target / 'original_hashes.json'
    if archive.exists() or manifest.exists():
        raise SystemExit('Preservation exists: refusing to overwrite originals')
    paths = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    if not paths:
        raise SystemExit('Original experiment directory is empty; refusing an empty preservation archive')
    files = {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in paths}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in files.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == record['sha256']
    source = [p for d in ('src', 'configs', 'tests', 'notebooks') for p in (ROOT/d).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_status_before_review': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
              'archive': archive.name, 'archive_sha256': sha(archive),
              'file_count': len(files), 'files': files,
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in source}}
    manifest.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k not in ('files', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
