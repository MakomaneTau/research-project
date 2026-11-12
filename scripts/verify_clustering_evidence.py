"""Reconcile retained annotation DB without claiming current access to raw CT.

Compares the actual installed pylidc state machine with custom clustering using
cached exact tree distances. Independently checks a deterministic distance sample
with chunked cdist. No training, new labels, or source artifacts are written.
"""
from pathlib import Path
import hashlib
import inspect
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lidc_experiment.compat import load_pylidc
from lidc_experiment.dataset import cluster_annotations

OUT=ROOT/'results_evidence/2026-11-12'
SRC=ROOT/'artifacts/chapter3'


def main():
    pl=load_pylidc();scans=pl.query(pl.Scan).all()
    audited=pd.read_csv(SRC/'data/audit_nodules.csv')
    exclusion=pd.read_csv(SRC/'data/exclusions.csv')
    ambiguous=set(exclusion.loc[exclusion.reason=='ambiguous_cluster','scan_id'])
    summaries=[];distances=[];selected=[];by_patient={}
    for scan in scans:by_patient.setdefault(scan.patient_id,[]).append(scan)
    eligible_count=0
    for i,(patient,series) in enumerate(sorted(by_patient.items())):
        candidates=[]
        for scan in series:
            spacing=float(scan.slice_spacing)
            if not np.isfinite(spacing) or not 0<spacing<=2.5:continue
            anns=sorted(scan.annotations,key=lambda a:a.id)
            points={a.id:np.asarray(a.contours_matrix,float) for a in anns}
            trees={k:cKDTree(v) for k,v in points.items()}
            cache={}
            for j,a in enumerate(anns):
                for b in anns[j+1:]:
                    cache[tuple(sorted((a.id,b.id)))]=float(trees[a.id].query(points[b.id],k=1,eps=0)[0].min())
            def metric(a,b):return cache[tuple(sorted((a.id,b.id)))]
            native=scan.cluster_annotations(metric=metric,verbose=False)
            custom,settings=cluster_annotations(scan)
            normalize=lambda groups:sorted(tuple(sorted(a.id for a in g)) for g in groups)
            same=normalize(native)==normalize(custom)
            summaries.append(dict(patient_id=patient,scan_id=scan.id,spacing=spacing,annotations=len(anns),clusters=len(custom),ambiguous_clusters=settings['ambiguous_clusters'],native_custom_match=same,final_tolerance=settings['tolerance_mm']))
            # Include every ambiguous series, plus regularly spaced deterministic
            # series; two distinct pairs per included series, no favorable choice.
            if cache and (scan.id in ambiguous or eligible_count%30==0):
                pairs=sorted(cache)
                chosen=list(dict.fromkeys([pairs[0],min(pairs,key=cache.get)]))
                for a,b in chosen:
                    p,q=points[a],points[b]
                    minimum=min(float(cdist(p[start:start+128],q).min()) for start in range(0,len(p),128))
                    distances.append(dict(scan_id=scan.id,annotation_a=a,annotation_b=b,points_a=len(p),points_b=len(q),tree_distance=cache[(a,b)],cdist_distance=minimum,absolute_difference=abs(cache[(a,b)]-minimum)))
            eligible_count+=1
            if any(2<=len(g)<=4 for g in custom):
                candidates.append((spacing,str(scan.series_instance_uid),scan,custom))
        if candidates:
            _,_,scan,groups=sorted(candidates,key=lambda v:v[:2])[0]
            for g in groups:
                if not 2<=len(g)<=4:continue
                if np.mean([a.diameter for a in g])<3:continue
                ids=tuple(sorted(a.id for a in g))
                selected.append(dict(patient_id=patient,scan_id=scan.id,annotation_ids=ids))
        if (i+1)%100==0:print(f'Clustering DB audit {i+1}/{len(by_patient)} patients',flush=True)
    a=pd.DataFrame(summaries);a.to_csv(OUT/'tables/clustering_equivalence.csv',index=False)
    b=pd.DataFrame(distances);b.to_csv(OUT/'tables/clustering_distance_sample.csv',index=False)
    actual={(int(r.scan_id),tuple(json.loads(r.annotation_ids))) for r in audited.itertuples()}
    rebuilt={(r['scan_id'],r['annotation_ids']) for r in selected}
    methods={'pylidc_cluster_annotations':inspect.getsource(pl.Scan.cluster_annotations),
             'pylidc_minimum_distance':inspect.getsource(pl.annotation_distance_metrics.pairdist),
             'contours_matrix':inspect.getsource(pl.Annotation.contours_matrix.fget),
             'contour_to_matrix':inspect.getsource(pl.Contour.to_matrix),
             'custom_cluster_annotations':inspect.getsource(cluster_annotations)}
    (OUT/'documents/clustering_implementation.txt').write_text('\n\n'.join(k+'\n'+v for k,v in methods.items()),encoding='utf-8')
    result=dict(source_patients=len(by_patient),source_series=len(scans),eligible_spacing_series=len(a),all_native_custom_groups_match=bool(a.native_custom_match.all()),distance_sample_pairs=len(b),maximum_distance_difference=float(b.absolute_difference.max()),audited_nodules=len(actual),reconstructed_nodules=len(rebuilt),audited_annotation_ids_match=actual==rebuilt,only_in_audit=[str(x) for x in sorted(actual-rebuilt)],only_in_reconstruction=[str(x) for x in sorted(rebuilt-actual)],ambiguous_clusters=int(a.ambiguous_clusters.sum()),raw_ct_revalidated=False,distance_coordinate_caveat='Installed pylidc passes contour index matrices directly to cdist; custom code preserves those coordinates. Neither implementation here converts x/y/k distances into a patient-space physical metric.',native_exception_caveat='Installed pylidc 0.2.3 breaks and returns unresolved groups; custom docstring claim that it raises is inaccurate for this installed version.')
    (OUT/'clustering_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
