"""Audit reconnected original CT storage without modifying it or fitting models."""
from pathlib import Path
from collections import Counter
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pydicom

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lidc_experiment.compat import load_pylidc
from lidc_experiment.dataset import _source_fingerprint, _load_scan, _prepare_record

OUT=ROOT/'results_evidence/2026-11-12'
SRC=ROOT/'artifacts/chapter3'


def main():
    config=json.loads((SRC/'run.json').read_text())['config']
    data=Path(config['data_root']);pl=load_pylidc()
    all_scans=pl.query(pl.Scan).all();scanmap={int(s.id):s for s in all_scans}
    db_patients={s.patient_id for s in all_scans}
    folders={p.name for p in data.iterdir() if p.is_dir()}
    inventory=[];other=[]
    for directory,_,files in os.walk(data):
        dicoms=[n for n in files if n.lower().endswith('.dcm')]
        if dicoms:
            p=Path(directory)
            try:
                ds=pydicom.dcmread(p/dicoms[0],stop_before_pixels=True)
                inventory.append(dict(patient_id=str(getattr(ds,'PatientID','')),path=str(p),modality=str(getattr(ds,'Modality','')),series_uid=str(getattr(ds,'SeriesInstanceUID','')),dicom_files=len(dicoms),bytes=sum((p/n).stat().st_size for n in dicoms)))
            except Exception as e:
                inventory.append(dict(path=str(p),error=str(e),dicom_files=len(dicoms)))
        for n in files:
            if not n.lower().endswith('.dcm'):other.append(dict(path=str(Path(directory)/n),bytes=(Path(directory)/n).stat().st_size))
    pd.DataFrame(inventory).to_csv(OUT/'tables/acquired_series_inventory.csv',index=False)
    pd.DataFrame(other,columns=['path','bytes']).to_csv(OUT/'tables/acquired_non_dicom_inventory.csv',index=False)
    frame=pd.read_csv(SRC/'data/all_nodules.csv')
    paths=[]
    for s in all_scans:
        try:
            p=Path(s.get_path_to_dicom_files())
            paths.append(dict(patient_id=s.patient_id,scan_id=s.id,series_uid=s.series_instance_uid,path=str(p),exists=p.is_dir(),dicom_files=sum(1 for n in p.iterdir() if n.suffix.lower()=='.dcm') if p.is_dir() else 0))
        except Exception as e:paths.append(dict(patient_id=s.patient_id,scan_id=s.id,series_uid=s.series_instance_uid,exists=False,error=str(e)))
    pd.DataFrame(paths).to_csv(OUT/'tables/database_raw_coverage.csv',index=False)
    start=time.perf_counter();records=[]
    for i,(scan_id,g) in enumerate(frame.groupby('scan_id',sort=True)):
        scan=scanmap[int(scan_id)];first=g.iloc[0]
        with np.load(first.npz_path,allow_pickle=False) as a:meta=json.loads(str(a['metadata'].item()))
        expected=meta['source_image_sha256'];saved_uids=set(json.loads(first.source_image_uids))
        path=Path(scan.get_path_to_dicom_files());actual={};bad=[];count=0;total=0
        fingerprint=_source_fingerprint(scan,config)
        for f in sorted(path.iterdir()):
            if f.suffix.lower()!='.dcm':continue
            blob=f.read_bytes();total+=len(blob)
            ds=pydicom.dcmread(io.BytesIO(blob),stop_before_pixels=True)
            if str(getattr(ds,'Modality',''))!='CT' or str(getattr(ds,'SeriesInstanceUID',''))!=str(scan.series_instance_uid):continue
            uid=str(ds.SOPInstanceUID);digest=hashlib.sha256(blob).hexdigest()
            # The saved decoder retains the least-InstanceNumber duplicate at
            # each z. Non-retained duplicate files have no frozen UID checksum.
            if uid in expected:
                actual[uid]=digest
                if digest!=expected[uid]:bad.append(uid)
            count+=1
        records.append(dict(scan_id=scan_id,patient_id=scan.patient_id,nodules=len(g),raw_files=count,retained_source_sops=len(expected),checked_bytes=total,all_expected_sops_found=set(actual)==set(expected)==saved_uids,sha256_mismatches=len(bad),fingerprint_matches=bool((g.source_fingerprint==fingerprint).all()),mismatched_sops=json.dumps(bad)))
        pd.DataFrame(records).to_csv(OUT/'tables/source_checksum_audit.csv',index=False)
        if (i+1)%10==0:print(f'Source hashes {i+1}/{frame.scan_id.nunique()} series; {sum(r["checked_bytes"] for r in records)/1e9:.2f} GB; {time.perf_counter()-start:.1f}s',flush=True)
    # Reconstruct all nine preselected development QA cases from original CT.
    panels=pd.read_csv(SRC/'data/source_panels/selected_cases.csv');fresh=[]
    for row in panels.itertuples():
        record=frame[frame.nodule_id==row.nodule_id].iloc[0].to_dict();scan=scanmap[int(row.scan_id)]
        raw,geometry=_load_scan(scan,config)
        regenerated=_prepare_record(record,config,None,scan,raw,geometry)
        with np.load(record['npz_path']) as a:
            identical={key:np.array_equal(a[key],regenerated['arrays'][key]) for key in ['hu','mask','image']}
        fresh.append(dict(nodule_id=row.nodule_id,scan_id=row.scan_id,partition=row.partition,**identical,selected_slice_matches=record['selected_slice_index']==regenerated['selected_slice_index']))
        print(f'Reconstructed development QA {row.nodule_id}',flush=True)
    pd.DataFrame(fresh).to_csv(OUT/'tables/raw_preprocessing_reconstruction.csv',index=False)
    exclusions=pd.read_csv(SRC/'data/exclusions.csv');failed=[]
    for scan_id,g in exclusions[exclusions.stage.isin(['geometry','preparation'])].groupby('scan_id'):
        try:_load_scan(scanmap[int(scan_id)],config);observed='unexpectedly passed'
        except Exception as e:observed=f'{type(e).__name__}: {e}'
        failed.append(dict(scan_id=scan_id,excluded_nodules=len(g),recorded_reason=g.detail.iloc[0],observed=observed,rejection_reproduced='Irregular slice positions or gantry shear' in observed))
    pd.DataFrame(failed).to_csv(OUT/'tables/geometry_exclusions_rechecked.csv',index=False)
    a=pd.DataFrame(records)
    summary=dict(raw_root=str(data),raw_patient_folders=len(folders),database_patients=len(db_patients),database_series=len(all_scans),folders_without_database_patient=sorted(folders-db_patients),database_patients_without_folder=sorted(db_patients-folders),acquired_series_directories=len(inventory),modality_counts=dict(Counter(r.get('modality','unreadable') for r in inventory)),all_database_series_accessible=all(r['exists'] for r in paths),prepared_series_checked=len(records),prepared_source_bytes_checked=sum(r['checked_bytes'] for r in records),source_sop_identities_match=bool(a.all_expected_sops_found.all()),source_sop_sha256_mismatches=int(a.sha256_mismatches.sum()),source_fingerprints_match=bool(a.fingerprint_matches.all()),qa_cases_reconstructed=len(fresh),all_qa_arrays_identical=all(all(r[k] for k in ['hu','mask','image','selected_slice_matches']) for r in fresh),geometry_exclusions_reproduced=all(r['rejection_reproduced'] for r in failed),retrieval_date_or_usage_terms_verified=False,scope='All prepared-series retained source SOP hashes; acquisition inventory is current local storage, not verified historical download provenance')
    (OUT/'source_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
