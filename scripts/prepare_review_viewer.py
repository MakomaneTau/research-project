"""Separate post-test interface-only amendment; no predictions are changed."""
from pathlib import Path
import hashlib
import html
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_evidence/2026-11-12'
SRC=ROOT/'artifacts/chapter3'


def main():
    page=OUT/'prototype/prepared_case_verified.html'
    text=page.read_text(encoding='utf8')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    models={p.name:digest(p) for p in (SRC/'training/models').iterdir() if p.suffix in ['.pt','.joblib']}
    record=dict(amendment='November verification interface copy only',config_hash=json.loads((SRC/'run.json').read_text())['config_hash'],frozen_manifest_sha256=digest(SRC/'frozen.json'),thresholds_sha256=digest(SRC/'development/thresholds.json'),cohort_sha256=digest(SRC/'cohort.csv'),models=models,original_frozen_viewer_unchanged=True,predictions_reused_without_modification=True)
    rows=''.join('<dt>'+html.escape(k)+'</dt><dd><code>'+html.escape(v)+'</code></dd>' for k,v in {**{k:v for k,v in record.items() if k.endswith('hash') or k.endswith('sha256')},**models}.items())
    section='<section aria-label="Experiment provenance"><h2>Experiment provenance</h2><p>Interface evidence copy prepared 12 November 2026 after test inspection. Predictions and operating thresholds are unchanged.</p><dl>'+rows+'</dl></section>'
    amended=(text.replace('</body>',section+'</body>') if '</body>' in text
             else text.replace('</html>',section+'</html>'))
    amended=amended.replace('</style>','code{overflow-wrap:anywhere;font-size:12px}dt{font-weight:600;margin-top:12px}dd{margin:4px 0 12px}</style>')
    path=OUT/'prototype/prepared_case_with_provenance.html';path.write_text(amended,encoding='utf8')
    assert all(v in amended for v in models.values()) and record['config_hash'] in amended
    assert text.split('<table',1)[1].split('</table>',1)[0] == amended.split('<table',1)[1].split('</table>',1)[0]
    (OUT/'prototype/interface_amendment.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(path)


if __name__=='__main__':main()
