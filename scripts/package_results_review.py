"""Check final package links/provenance and bundle review, preserving original ZIP."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_evidence/2026-11-12'


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    original=json.loads((OUT/'preservation/original_hashes.json').read_text())
    assert all(sha(ROOT/p)==r['sha256'] for p,r in original['files'].items())
    assert all(sha(ROOT/p)==h for p,h in original['source_sha256'].items())
    assert sha(OUT/'preservation/chapter3_original.zip')==original['archive_sha256']
    docs=json.loads((OUT/'documents/source_hashes.json').read_text())
    assert all(sha(Path(p))==r['sha256'] for p,r in docs.items())
    required=['audit_summary.json','clustering_audit.json','representation_audit.json','source_audit.json']
    audits={n:json.loads((OUT/n).read_text()) for n in required}
    assert audits['audit_summary.json']['execution_complete'] and audits['audit_summary.json']['failed']==0
    assert audits['clustering_audit.json']['all_native_custom_groups_match'] and audits['clustering_audit.json']['audited_annotation_ids_match']
    assert audits['representation_audit.json']['all_original_features_match'] and audits['representation_audit.json']['all_mask_predictions_match'] and audits['representation_audit.json']['permutation_results_match']
    source=audits['source_audit.json']
    assert source['source_sop_sha256_mismatches']==0 and source['source_sop_identities_match'] and source['all_qa_arrays_identical'] and source['geometry_exclusions_reproduced']
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    repository_hashes={p:sha(ROOT/p) for p in tracked}
    snapshot=OUT/'preservation/repository_snapshot.zip'
    with zipfile.ZipFile(snapshot,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in tracked:z.write(ROOT/p,p)
    with zipfile.ZipFile(snapshot) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in repository_hashes.items())
    (OUT/'preservation/repository_snapshot_hashes.json').write_text(json.dumps(repository_hashes,indent=2),encoding='utf8')
    links=[]
    for file in OUT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf8')):
            if target.startswith(('https:','http:','#')):continue
            target=target.split('#')[0]
            assert (file.parent/target).exists(),f'Missing link {file.name}: {target}'
            links.append(dict(document=file.name,target=target))
    summary=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),original_artifact_files_unchanged=len(original['files']),original_source_files_unchanged=len(original['source_sha256']),repository_snapshot_files=len(tracked),repository_snapshot_sha256=sha(snapshot),dissertation_and_input_documents_unchanged=True,archive_sha256_unchanged=True,required_audits_complete=True,local_markdown_links_checked=len(links),review_is_post_test_inspection=True,experiment_refits=0,browser_rendering_verified=False,missing_approval_acquisition_documents_left_unresolved_at_user_request=True)
    (OUT/'package_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    excluded={'review_package.zip','package_files_sha256.json','review_package_sha256.txt'}
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in excluded]
    hashes={p.relative_to(OUT).as_posix():sha(p) for p in files}
    (OUT/'package_files_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    target=OUT/'review_package.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[OUT/'package_files_sha256.json']:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p,digest in hashes.items():assert hashlib.sha256(z.read(p)).hexdigest()==digest
    (OUT/'review_package_sha256.txt').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps(dict(summary,review_zip_bytes=target.stat().st_size,review_zip_sha256=sha(target)),indent=2))


if __name__=='__main__':main()
