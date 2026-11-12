"""Independent saved-evidence audit; never fits models or writes chapter3.

Run with the project's Python 3.11 environment. Outputs are additional analyses
performed after test inspection, not replacement primary experiment artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SRC = ROOT / 'artifacts/chapter3'
OUT = ROOT / 'results_evidence/2026-11-12'
TABLES = OUT / 'tables'
CHECKS = []
DISCREPANCIES = []
COMPLETED = False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(name):
    return json.loads((SRC / name).read_text(encoding='utf-8'))


def csv(name):
    return pd.read_csv(SRC / name)


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, (np.integer, np.bool_)): return x.item()
        if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
        return x
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / name, index=False)
    return frame


def check(name, passed, detail='', scope='recomputed'):
    CHECKS.append(dict(check=name, passed=bool(passed), scope=scope, detail=str(detail)))
    if not passed:
        DISCREPANCIES.append(dict(item=name, detail=str(detail), impact='review required'))


def compare(name, fresh, old, keys, tolerance=1e-12):
    both = fresh.merge(old, on=keys, suffixes=('_new', '_saved'), validate='one_to_one')
    check(name + ': row keys', len(both) == len(fresh) == len(old), f'{len(both)} paired rows')
    for col in fresh:
        if col in keys or col not in old or not pd.api.types.is_numeric_dtype(fresh[col]):
            continue
        a = both[col + '_new'].to_numpy(float)
        b = both[col + '_saved'].to_numpy(float)
        delta = np.abs(a - b)
        passed = np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True)
        check(name + ': ' + col, passed, f'max absolute difference {np.nanmax(delta) if len(delta) and np.isfinite(delta).any() else 0:.4g}; tolerance {tolerance}')


def measures(y, score, probability, threshold, weights=None):
    """Independent formulas; original evaluation functions are not called."""
    y, score, probability = np.asarray(y, int), np.asarray(score, float), np.asarray(probability, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    positive = score >= threshold
    tp, fn, tn, fp = [w[m].sum() for m in ((y == 1) & positive, (y == 1) & ~positive,
                                           (y == 0) & ~positive, (y == 0) & positive)]
    def divide(a, b): return a / b if b else np.nan
    sen, spe = divide(tp, tp + fn), divide(tn, tn + fp)
    return dict(n_nodules=len(y), positive_nodules=int(y.sum()), prevalence=np.average(y, weights=w),
                threshold=threshold, tp=tp, fn=fn, tn=tn, fp=fp,
                roc_auc=roc_auc_score(y, score, sample_weight=w) if len(np.unique(y)) == 2 else np.nan,
                average_precision=average_precision_score(y, score, sample_weight=w) if y.any() else np.nan,
                accuracy=divide(tp + tn, w.sum()), balanced_accuracy=(sen + spe) / 2,
                sensitivity=sen, specificity=spe, precision=divide(tp, tp + fp),
                f1=divide(2 * tp, 2 * tp + fp + fn), brier_score=np.average((probability - y) ** 2, weights=w))


def threshold(y, score):
    # With >= and a fixed sensitivity floor, the largest feasible observed
    # threshold also maximizes specificity among feasible score cutoffs.
    positives = np.sort(np.asarray(score)[np.asarray(y) == 1])[::-1]
    return float(positives[int(np.ceil(.9 * len(positives))) - 1])


def preservation():
    original = json.loads((OUT / 'preservation/original_hashes.json').read_text())
    check('Original artifact bytes unchanged at audit start', all((ROOT/p).is_file() and sha(ROOT/p) == d['sha256'] for p,d in original['files'].items()), len(original['files']))
    rows = []
    for name in ('frozen.json', 'completion_manifest.json'):
        record = read(name)
        for path, digest in record.get('files', record.get('files_sha256', {})).items():
            p = Path(path)
            actual = sha(p) if p.is_file() else None
            rows.append(dict(manifest=name, path=path, expected=digest, actual=actual, matches=actual == digest))
    frame = table('historical_hash_verification.csv', rows)
    for name, group in frame.groupby('manifest'):
        check(name + ' hashes', group.matches.all(), f'{group.matches.sum()}/{len(group)}')
    expected = read('completion_manifest.json').get('frozen_manifest_sha256')
    check('Completion links frozen manifest', sha(SRC/'frozen.json') == expected)
    # Each inventory status names the verification boundary. A byte match does
    # not certify scientific content or physical CT alignment.
    inventory = []
    for path, d in original['files'].items():
        part = Path(path).relative_to('artifacts/chapter3').parts[0]
        inventory.append(dict(path=path, category=part, bytes=d['bytes'], sha256=d['sha256'],
                              status='verified', verification_scope='existence, preserved archive and SHA-256; scientific checks are listed separately'))
    table('evidence_inventory.csv', inventory)
    docs = [Path(r'C:\Users\ponts\OneDrive - Eduvos\Honours\Block 4\ITRPA')/n
            for n in ['Revised Draft.pdf','Ruubric.pdf','Template.pdf']]
    docs.append(Path(r'C:\Users\ponts\Downloads\Research_Project_Updated_Handover_2026-11-12.md'))
    save('documents/source_hashes.json', {str(p): {'sha256': sha(p), 'bytes':p.stat().st_size} for p in docs})
    shutil.copyfile(docs[-1], OUT/'documents'/docs[-1].name)
    notebook = json.loads((ROOT/'notebooks/chapter3_experiment.ipynb').read_text(encoding='utf-8'))
    cells = []
    for i, cell in enumerate(notebook['cells']):
        outputs = cell.get('outputs', [])
        cells.append(dict(cell=i, kind=cell['cell_type'], execution_count=cell.get('execution_count'),
                          errors=sum(o.get('output_type') == 'error' for o in outputs), outputs=len(outputs)))
    table('notebook_cells.csv', cells)
    check('Saved notebook has no error outputs', all(r['errors'] == 0 for r in cells), 'saved output inspection, not notebook rerun')


def cohort_audit():
    from lidc_experiment.splitting import assign_partitions
    cohort, prepared, audited, exclusions = csv('cohort.csv'), csv('data/all_nodules.csv'), csv('data/audit_nodules.csv'), csv('data/exclusions.csv')
    features, common = csv('features/radiomics_features.csv'), csv('common_all_labels.csv')
    raw = read('data/audit_summary.json')
    rows = [dict(stage='annotation database collection', patients=raw['source_patients'], series=raw['source_scans'], nodules=None, unit_note='database coverage, not a download manifest')]
    for name, f in [('reader/size/series eligible', audited), ('prepared', prepared), ('radiomics common all labels', common), ('mean intermediate excluded', common[common.label == -1]), ('primary binary',cohort)]:
        rows.append(dict(stage=name,patients=f.patient_id.nunique(), series=f.scan_id.nunique(), nodules=len(f),unit_note='retained counts; rows are not additive across stages'))
    table('cohort_flow.csv', rows)
    e = exclusions.groupby(['stage','reason'],dropna=False).agg(events=('patient_id','size'),patients=('patient_id','nunique'),series=('scan_id','nunique'),nodule_keys=('nodule_id','nunique')).reset_index()
    e['counting_unit'] = np.where(e.stage=='series','series exclusion event','nodule exclusion event')
    table('exclusions_by_unit.csv',e)
    check('Series count reconciliation', raw['source_scans'] == audited.scan_id.nunique() + len(exclusions[exclusions.stage=='series']), '1018 = selected eligible series + series exclusion events')
    failed = exclusions[exclusions.stage.isin(['geometry','preparation'])]
    check('Prepared nodule reconciliation', set(audited.nodule_id) == set(prepared.nodule_id)|set(failed.nodule_id) and len(audited) == len(prepared)+len(failed))
    check('Feature extraction complete', set(features.nodule_id)==set(prepared.nodule_id)==set(common.nodule_id), f'{len(features)} rows; no missing prepared feature cases')
    check('Primary binary common cohort', set(cohort.nodule_id)==set(common.loc[common.label.isin([0,1]),'nodule_id']))
    expected = []
    for row in prepared.itertuples():
        ratings = np.array(json.loads(row.ratings))
        mean = ratings.mean()
        label = 0 if mean <= 2 else 1 if mean >= 4 else -1
        expected.append(label)
        assert row.reader_count == len(ratings) and len(ratings)>=2 and len(ratings)<=4
        assert abs(mean-row.rating_mean)<1e-12 and abs(np.median(ratings)-row.rating_median)<1e-12
    check('Labels and reader vectors reconstructed', np.array_equal(expected,prepared.label), f'{len(prepared)} rows')
    check('1594 minus 1049 equals 545', len(prepared) - int((prepared.label == -1).sum()) == len(cohort))
    partitions=[]
    for name,g in cohort.groupby('partition'):
        partitions.append(dict(partition=name,patients=g.patient_id.nunique(),series=g.series_uid.nunique(),nodules=len(g),low=int((g.label==0).sum()),high=int((g.label==1).sum()),prevalence=g.label.mean(),high_patient_stratum=g.groupby('patient_id').label.max().sum()))
    table('class_distribution.csv',partitions)
    contributions=cohort.groupby(['partition','patient_id']).label.agg(nodules='size',high='sum').reset_index()
    contributions['low']=contributions.nodules-contributions.high
    table('patient_nodule_contributions.csv',contributions)
    table('training_fold_distribution.csv',[dict(fold=f,patients=g.patient_id.nunique(),nodules=len(g),low=int((g.label==0).sum()),high=int((g.label==1).sum())) for f,g in cohort[cohort.partition=='train'].groupby('fold')])
    leakage=[]
    for key in ['patient_id','series_uid','nodule_id']:
        count=int(cohort.groupby(key).partition.nunique().gt(1).sum())
        leakage.append(dict(identifier=key,overlap_count=count,scope='between partitions'))
        check('No '+key+' overlap', count==0)
    expanded = cohort[['partition','source_image_uids']].copy()
    expanded['source_image_uids']=expanded.source_image_uids.map(json.loads)
    expanded=expanded.explode('source_image_uids')
    count=int(expanded.groupby('source_image_uids').partition.nunique().gt(1).sum())
    leakage.append(dict(identifier='source_image_uids',overlap_count=count,scope=f'{expanded.source_image_uids.nunique()} unique source SOP identities, entire selected series lists'))
    check('No source image overlap',count==0)
    train=cohort[cohort.partition=='train']
    count=int(train.groupby('patient_id').fold.nunique().gt(1).sum())
    leakage.append(dict(identifier='patient_id',overlap_count=count,scope='between training folds'))
    check('No training fold patient overlap',count==0)
    table('leakage_audit.csv',leakage)
    repeat=assign_partitions(cohort.drop(columns=['partition','fold']),seed=42).set_index('nodule_id')
    check('Split and folds independently regenerated', repeat[['partition','fold']].equals(cohort.set_index('nodule_id').loc[repeat.index,['partition','fold']]))
    image_checks=[]
    for row in prepared.itertuples():
        with np.load(row.npz_path,allow_pickle=False) as a:
            im,mask,hu=a['image'],a['mask'],a['hu']
            m=json.loads(str(a['metadata'].item()))
            valid=(im.shape==(2,64,64) and mask.shape==hu.shape==(64,64) and np.isfinite(im).all()
                   and np.isfinite(hu).all() and np.isin(mask,[0,1]).all() and mask.any()
                   and np.array_equal(im[1],mask) and np.allclose(im[0],((hu+1000)/1400)*mask,rtol=0,atol=1e-7)
                   and hu.min()>=-1000 and hu.max()<=400 and m['nodule_id']==row.nodule_id
                   and m['preprocessing_hash']==row.preprocessing_hash and m['source_fingerprint']==row.source_fingerprint
                   and row.selected_slice_index==m['selected_slice_index'] and abs(row.lesion_area_mm2-mask.sum())<1e-10)
            image_checks.append(dict(nodule_id=row.nodule_id,passed=valid,mask_pixels=int(mask.sum()),mask_touches_crop_edge=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())))
    im=table('prepared_array_audit.csv',image_checks)
    check('Prepared arrays, labels and metadata valid',im.passed.all(),f'{len(im)} cached cases; source physical alignment requires original CT')
    median=csv('secondary/median_label_changes.csv')
    check('Median task excludes test patients',not(set(median.patient_id)&set(cohort.loc[cohort.partition=='test','patient_id'])))
    alternative=common.merge(cohort.loc[cohort.partition!='test',['patient_id','partition','fold']].drop_duplicates(),on='patient_id',validate='many_to_one')
    alternative['label']=np.where(alternative.rating_median<=2,0,np.where(alternative.rating_median>=4,1,-1)).astype('int64')
    matched=alternative.set_index('nodule_id').sort_index()[['label','partition','fold']]
    check('Median alternative task independently reconstructed',matched.equals(median.set_index('nodule_id').sort_index()[['label','partition','fold']]))
    table('median_cohort_changes.csv',[dict(partition=p,all_development_nodules=len(g),mean_binary=int(g.primary_mean_label.isin([0,1]).sum()),median_binary=int(g.label.isin([0,1]).sum()),newly_eligible=int(((g.primary_mean_label==-1)&g.label.isin([0,1])).sum()),no_longer_eligible=int((g.primary_mean_label.isin([0,1])&(g.label==-1)).sum()),changed_low_high=int((g.primary_mean_label.isin([0,1])&g.label.isin([0,1])&(g.primary_mean_label!=g.label)).sum())) for p,g in median.groupby('partition')])
    return cohort,features


def training_audit(cohort,features):
    from lidc_experiment.training import load_model, predict_model, RadiomicsPreprocessor, _feature_frame
    import torch
    selection=read('training/selection.json'); config=read('run.json')['config']
    train=cohort[cohort.partition=='train']; cv=csv('training/cv_results.csv')
    check('All 144 primary candidate folds succeeded',len(cv)==144 and (cv.status=='success').all())
    model_rows=[]; cv_rows=[]; histories=[]; candidates=[]; models={}
    for (name,candidate), group in cv.groupby(['model','candidate'],sort=False):
        candidates.append(dict(model=name,candidate=int(candidate),settings=group.settings.iloc[0],mean_auc=group.auc.mean(),fold_0_auc=group.loc[group.fold==0,'auc'].iloc[0],fold_1_auc=group.loc[group.fold==1,'auc'].iloc[0],fold_2_auc=group.loc[group.fold==2,'auc'].iloc[0]))
        for row in group.itertuples():
            folder=SRC/'training/cv_fits'/name/f'candidate_{candidate:02d}'/f'fold_{row.fold}'
            held=train[train.fold==row.fold]; fit=train[train.fold!=row.fold]
            pred=pd.read_csv(folder/'predictions.csv')
            meta=json.loads((folder/('model.pt.metadata.json' if name=='cnn' else 'model.joblib.metadata.json')).read_text())
            ok=(set(pred.nodule_id)==set(held.nodule_id) and set(meta['fit_nodule_ids'])==set(fit.nodule_id)
                and set(meta['fit_patient_ids'])==set(fit.patient_id) and abs(roc_auc_score(pred.label,pred.score)-row.auc)<1e-12)
            key=held.set_index('nodule_id').loc[pred.nodule_id]
            ok=ok and np.array_equal(key.label,pred.label) and np.array_equal(key.patient_id,pred.patient_id)
            if name=='cnn':
                early=json.loads((folder/'early_stopping.json').read_text())
                a,b,c=map(set,[early['fit_patient_ids'],early['early_stopping_patient_ids'],early['outer_held_out_patient_ids']])
                ok=ok and not(a&b or a&c or b&c) and a|b==set(fit.patient_id) and c==set(held.patient_id)
                best_auc=-np.inf; best_epoch=0; stale=0
                for h in early['history']:
                    if h['inner_validation_auc']>=best_auc+.0001:
                        best_auc=h['inner_validation_auc'];best_epoch=h['epoch'];stale=0
                    else: stale+=1
                    histories.append(dict(candidate=candidate,fold=row.fold,**h))
                ok=ok and best_epoch==early['best_epoch']==row.best_epoch and (stale==8 or early['epochs_run']==60)
            else:
                bundle=load_model(folder/'model.joblib')
                processor=bundle.feature_processor
                expected=RadiomicsPreprocessor(scale=name!='random_forest').fit(_feature_frame(fit,features))
                ok=ok and set(processor.fit_nodule_ids_)==set(fit.nodule_id) and set(processor.fit_patient_ids_)==set(fit.patient_id)
                ok=ok and processor.retained_features_==expected.retained_features_ and np.allclose(processor.medians_,expected.medians_,rtol=0,atol=1e-12)
            cv_rows.append(dict(model=name,candidate=candidate,fold=row.fold,passed=ok,auc=row.auc,fit_nodules=len(fit),held_nodules=len(held),retained_features=len(meta.get('retained_features',[]))))
    cands=table('complete_candidates.csv',candidates)
    for name in selection['models']:
        group=cands[cands.model==name].sort_values('candidate'); high=group.mean_auc.max()
        winner=group[group.mean_auc>=high-.0001].iloc[0]
        check(name+' selection rule',int(winner.candidate)==selection['models'][name]['candidate_index'])
    table('cv_provenance_audit.csv',cv_rows); table('cnn_inner_histories.csv',histories)
    check('CV provenance, preprocessing, scores and stopping',all(x['passed'] for x in cv_rows),f'{len(cv_rows)} fold artifacts')
    native_device='cuda' if torch.cuda.is_available() else 'cpu'
    config=copy.deepcopy(config);config['runtime']['device']=native_device
    torch.set_num_threads(1)
    reload_rows=[]
    for name,spec in selection['models'].items():
        path=SRC/'training/models'/(name+('.pt' if name=='cnn' else '.joblib'))
        model=load_model(path);models[name]=model;meta=model.metadata
        check(name+' final training IDs',set(meta['fit_patient_ids'])==set(train.patient_id) and set(meta['fit_nodule_ids'])==set(train.nodule_id))
        check(name+' final selected settings',meta['settings']==spec['settings'])
        if name!='cnn':
            fp=model.feature_processor
            expected=RadiomicsPreprocessor(scale=name!='random_forest').fit(_feature_frame(train,features))
            check(name+' final preprocessing',fp.retained_features_==expected.retained_features_ and np.allclose(fp.medians_,expected.medians_,rtol=0,atol=1e-12) and np.allclose(fp.means_,expected.means_,rtol=0,atol=1e-12) and np.allclose(fp.scales_,expected.scales_,rtol=0,atol=1e-12) and set(fp.fit_nodule_ids_)==set(train.nodule_id))
        if name=='svm':
            for f in meta['calibration_folds']:
                fit_ids=set(train.loc[train.fold!=f['fold'],'patient_id']);held_ids=set(train.loc[train.fold==f['fold'],'patient_id'])
                check('SVM calibration fold '+str(f['fold']),set(f['fit_patient_ids'])==fit_ids and set(f['held_out_patient_ids'])==held_ids and not fit_ids&held_ids)
            oof=pd.DataFrame(meta['calibration_oof'])
            check('SVM calibration OOF covers train only',set(oof.nodule_id)==set(train.nodule_id) and len(oof)==len(train) and np.array_equal(train.set_index('nodule_id').loc[oof.nodule_id,'label'],oof.label))
        for partition,file in [('test','results/test_predictions.csv'),('validation','development/validation_predictions.csv')]:
            subset=cohort[cohort.partition==partition]
            new=predict_model(model,subset,features,config).set_index('nodule_id')
            old=csv(file);old=old[old.model==name].set_index('nodule_id').loc[new.index]
            for field in ['score','probability']:
                delta=float(np.max(np.abs(new[field]-old[field])))
                ok=np.allclose(new[field],old[field],rtol=1e-6,atol=1e-7)
                reload_rows.append(dict(model=name,partition=partition,field=field,max_absolute_difference=delta,passed=ok,device=native_device if name=='cnn' else 'cpu'))
                check(f'{name} {partition} reload {field}',ok,f'max difference {delta}; rtol=1e-6, atol=1e-7')
        model_rows.append(dict(model=name,selected_candidate=spec['candidate_index'],settings=json.dumps(spec['settings']),fold_aucs=json.dumps(spec['fold_aucs']),mean_cv_auc=spec['mean_auc'],retained_features=len(meta.get('retained_features',[])),final_epochs=meta.get('final_epochs'),fold_best_epochs=json.dumps(spec.get('fold_best_epochs')),fit_wall_seconds=meta['fit_wall_seconds'],search_wall_seconds=selection['search_wall_seconds'][name],device=meta.get('device','cpu'),precision=meta['numerical_precision'],seed=meta['seed'],warnings=json.dumps(meta['warnings']),peak_process_rss_bytes=meta['peak_process_rss_bytes'],peak_cuda_allocated_bytes=meta.get('peak_cuda_allocated_bytes'),model_bytes=path.stat().st_size,implementation_sha256=meta['implementation_sha256'],current_implementation_sha256=sha(ROOT/'src/lidc_experiment/training.py')))
    table('model_selection_and_training.csv',model_rows);table('reload_predictions_audit.csv',reload_rows)
    save('audit_runtime.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),cuda_available=torch.cuda.is_available(),reload_device=native_device,packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch','pylidc','pyradiomics']},no_model_fitting=True))
    return models,config


def predictive_audit(cohort):
    data=csv('results/test_predictions.csv'); validation=csv('development/validation_predictions.csv'); thresholds=read('development/thresholds.json')
    threshold_rows=[]
    for name,g in validation.groupby('model'):
        t=threshold(g.label,g.score)
        threshold_rows.append(dict(model=name,saved_threshold=thresholds[name],recalculated_threshold=t,validation_sensitivity=np.mean(g.loc[g.label==1,'score']>=t),validation_nodules=len(g)))
        check(name+' validation threshold',abs(t-thresholds[name])<1e-14)
    table('threshold_verification.csv',threshold_rows)
    by={name:g.sort_values('nodule_id').reset_index(drop=True) for name,g in data.groupby('model',sort=True)}
    test=cohort[cohort.partition=='test'].set_index('nodule_id')
    rows=[];prob=[];patient=[];dist=[];sub=[];cal=[]
    for name,g in by.items():
        check(name+' complete paired 90 test predictions',len(g)==90 and not g.nodule_id.duplicated().any() and set(g.nodule_id)==set(test.index) and np.array_equal(test.loc[g.nodule_id,'patient_id'],g.patient_id) and np.array_equal(test.loc[g.nodule_id,'label'],g.label))
        rows.append(dict(model=name,n_patients=g.patient_id.nunique(),**measures(g.label,g.score,g.probability,thresholds[name])))
        prob.append(dict(model=name,**measures(g.label,g.probability,g.probability,.5)))
        weight=1/g.groupby('patient_id').nodule_id.transform('count')
        patient.append(dict(model=name,**measures(g.label,g.score,g.probability,thresholds[name],weight)))
        for partition,values in [('test',g),('validation',validation[validation.model==name])]:
            dist.append(dict(model=name,partition=partition,score_min=values.score.min(),score_max=values.score.max(),probability_min=values.probability.min(),probability_q25=values.probability.quantile(.25),probability_median=values.probability.median(),probability_q75=values.probability.quantile(.75),probability_max=values.probability.max(),probability_std=values.probability.std(),probability_distinct=values.probability.nunique(),probability_050_positive=int((values.probability>=.5).sum())))
        for i,inds in enumerate(np.array_split(np.argsort(g.probability.to_numpy(),kind='stable'),5)):
            a=g.iloc[inds]
            cal.append(dict(model=name,bin=i+1,n_nodules=len(a),mean_probability=a.probability.mean(),observed_fraction=a.label.mean()))
        for variable,bins,labels in [('diameter_mm',[-np.inf,6,10,np.inf],['<=6','>6 to 10','>10']),('rating_range',[-np.inf,1,np.inf],['0-1','>=2'])]:
            for key,a in g.groupby(pd.cut(g[variable],bins,labels=labels),observed=True):
                sub.append(dict(model=name,variable=variable,subgroup=str(key),n_patients=a.patient_id.nunique(),**measures(a.label,a.score,a.probability,thresholds[name])))
    for file,values,keys in [('metrics.csv',rows,['model']),('metrics_probability_050.csv',prob,['model']),('metrics_patient_weighted.csv',patient,['model']),('subgroup_metrics.csv',sub,['model','variable','subgroup']),('calibration_bins.csv',cal,['model','bin'])]:
        frame=table(file,values);compare(file,frame,csv('results/'+file),keys)
    table('prediction_distributions.csv',dist)
    # Independent patient draw implementation, retaining every draw's multiplicity.
    base=next(iter(by.values()));patients=np.unique(base.patient_id);members=[np.flatnonzero(base.patient_id.to_numpy()==p) for p in patients]
    rng=np.random.default_rng(42);draws=[];samples=[];valid=attempts=rejected=0
    metric_names=['roc_auc','average_precision','accuracy','balanced_accuracy','sensitivity','specificity','precision','f1','brier_score']
    while valid<2000 and attempts<20000:
        attempts+=1;draw=rng.integers(0,len(patients),size=len(patients));inds=np.concatenate([members[j] for j in draw])
        if base.label.iloc[inds].nunique()<2: rejected+=1;continue
        draws.append(draw);results={}
        for name,g in by.items():
            a=g.iloc[inds];m=measures(a.label,a.score,a.probability,thresholds[name]);results[name]=m
            samples.append(dict(replicate=valid,model=name,**{key:m[key] for key in metric_names}))
        valid+=1
    samples=table('bootstrap_replicates.csv',samples)
    np.savez_compressed(OUT/'tables/bootstrap_patient_draws.npz',patients=patients.astype(str),draws=np.array(draws),seed=42)
    intervals=[]
    for name,g in samples.groupby('model'):
        for metric in metric_names:
            a=g[metric].to_numpy();a=a[np.isfinite(a)]
            bounds=np.percentile(a,[2.5,97.5]) if len(a) else [np.nan,np.nan]
            intervals.append(dict(model=name,metric=metric,estimate=next(r[metric] for r in rows if r['model']==name),lower_95=bounds[0],upper_95=bounds[1],valid_replicates=len(a)))
    fresh=table('metric_intervals.csv',intervals)
    old=csv('results/metric_intervals.csv');compare('bootstrap intervals',fresh,old,['model','metric'])
    comparator=read('training/selection.json')['selected_traditional'];diff=[]
    for metric in ['roc_auc','sensitivity','specificity']:
        wide=samples.pivot(index='replicate',columns='model',values=metric)
        delta=wide.cnn-wide[comparator];q=np.percentile(delta,[2.5,97.5])
        diff.append(dict(metric=metric,estimate=next(r[metric] for r in rows if r['model']=='cnn')-next(r[metric] for r in rows if r['model']==comparator),lower_95=q[0],upper_95=q[1]))
    table('primary_comparison.csv',diff)
    save('bootstrap_audit.json',dict(seed=42,requested=2000,valid=valid,attempts=attempts,rejected_single_class=rejected,comparator=comparator,repeated_patients_preserved=True,draw_array='tables/bootstrap_patient_draws.npz',differences=diff,uncertainty='patient sampling conditional on fixed fitted models, selected settings and thresholds'))
    saved=read('results/primary_comparison.json')
    check('Bootstrap attempts and rejection counts',valid==saved['valid_replicates'] and attempts==saved['attempts'] and rejected==saved['rejected_single_class'])
    for a in diff:
        b=saved['differences'][a['metric']]
        for k in ['estimate','lower_95','upper_95']:
            check('Primary difference '+a['metric']+' '+k,abs(a[k]-b[k])<1e-12,f'{a[k]} vs {b[k]}')
    # Plotting is independent output only; input remains preserved predictions.
    from lidc_experiment.evaluation import _plots
    _plots(data,thresholds,OUT/'figures/predictive')


def secondary_audit(cohort,models,config):
    from lidc_experiment.explanations import grad_cam, select_error_panel
    from lidc_experiment.training import load_model
    import torch
    secondary=csv('secondary/development_metrics.csv');rows=[]
    for row in secondary.itertuples():
        path=SRC/'secondary/fits'/row.analysis/(row.detail if isinstance(row.detail,str) else '')/row.model
        p=pd.read_csv(path/'validation_predictions.csv')
        model=load_model(path/('model.pt' if row.model=='cnn' else 'model.joblib'))
        valid=(len(p)==row.validation_nodules and abs(roc_auc_score(p.label,p.score)-row.roc_auc)<1e-12
               and abs(average_precision_score(p.label,p.score)-row.average_precision)<1e-12
               and not(set(model.metadata['fit_patient_ids'])&set(p.patient_id))
               and not(set(model.metadata['fit_patient_ids'])&set(cohort.loc[cohort.partition=='test','patient_id']))
               and not(set(p.patient_id)&set(cohort.loc[cohort.partition=='test','patient_id'])))
        rows.append(dict(analysis=row.analysis,detail=row.detail,model=row.model,seed=row.seed,passed=valid,roc_auc=row.roc_auc,average_precision=row.average_precision,fit_seconds=row.fit_wall_seconds))
    table('secondary_verification.csv',rows)
    check('29 development-only secondary fit records',len(rows)==29 and all(x['passed'] for x in rows))
    table('robustness_development.csv',secondary)
    mask=csv('secondary/mask_sensitivity.csv');check('Mask score changes arithmetic',np.allclose(mask.alternative_score-mask.original_score,mask.score_change,atol=1e-12,rtol=0))
    table('mask_robustness_summary.csv',[dict(model=name,variant=variant,rows=len(g),cases=g.nodule_id.nunique(),absolute_change_median=g.score_change.abs().median(),absolute_change_q25=g.score_change.abs().quantile(.25),absolute_change_q75=g.score_change.abs().quantile(.75),absolute_change_max=g.score_change.abs().max(),slice_changes=int((g.original_slice!=g.alternative_slice).sum()),score_scale='native SVM decision' if name=='svm' else 'high-suspicion probability') for (name,variant),g in mask.groupby(['model','variant'])])
    coef=csv('explanations/logistic_coefficients.csv').set_index('feature');lr=models['logistic_regression']
    check('LR coefficient names and directions',np.allclose(coef.loc[lr.retained_features,'standardized_coefficient'],lr.estimator.coef_.ravel(),atol=1e-12,rtol=0))
    table('logistic_coefficients.csv',coef.reset_index())
    table('permutation_importance.csv',csv('explanations/permutation_importance.csv'))
    sanity=csv('explanations/parameter_randomization_sanity.csv');cnn=models['cnn'].network.cpu().eval()
    with torch.random.fork_rng():
        torch.manual_seed(42);random_network=copy.deepcopy(cnn)
        for module in random_network.modules():
            if hasattr(module,'reset_parameters'):module.reset_parameters()
    random_network.eval();new=[]
    for row in sanity.itertuples():
        record=cohort.set_index('nodule_id').loc[row.nodule_id]
        with np.load(record.npz_path) as a: image=a['image']
        cam,logit=grad_cam(cnn,image);random_cam,random_logit=grad_cam(random_network,image)
        corr=np.corrcoef(cam.ravel(),random_cam.ravel())[0,1] if cam.std() and random_cam.std() else np.nan
        new.append(dict(nodule_id=row.nodule_id,trained_logit=logit,random_logit=random_logit,map_mean_absolute_difference=float(np.abs(cam-random_cam).mean()),map_pearson_correlation=corr,trained_map_constant=not bool(cam.std()),random_map_constant=not bool(random_cam.std())))
    frame=table('randomization_recomputed_cpu.csv',new)
    # CPU parameter initialization uses a different RNG stream from CUDA.
    # Keep this supplemental observation; compare the original check on its
    # actual device below rather than treating different random weights as a
    # numerical prediction tolerance failure.
    if torch.cuda.is_available():
        cnn=cnn.cuda().eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42);random_network=copy.deepcopy(cnn)
            for module in random_network.modules():
                if hasattr(module,'reset_parameters'):module.reset_parameters()
        random_network.eval();same_device=[]
        for row in sanity.itertuples():
            record=cohort.set_index('nodule_id').loc[row.nodule_id]
            with np.load(record.npz_path) as a: image=a['image']
            cam,logit=grad_cam(cnn,image);random_cam,random_logit=grad_cam(random_network,image)
            corr=np.corrcoef(cam.ravel(),random_cam.ravel())[0,1] if cam.std() and random_cam.std() else np.nan
            same_device.append(dict(nodule_id=row.nodule_id,trained_logit=logit,random_logit=random_logit,map_mean_absolute_difference=float(np.abs(cam-random_cam).mean()),map_pearson_correlation=corr,trained_map_constant=not bool(cam.std()),random_map_constant=not bool(random_cam.std())))
        frame=table('randomization_recomputed_original_cuda.csv',same_device)
        compare('Grad-CAM original CUDA reproduction',frame,sanity,['nodule_id'],tolerance=2e-5)
        save('randomization_device_note.json',dict(resolution='CPU and CUDA reset_parameters use different seeded random streams. Original randomisation check used a CUDA-resident network; repeat on CUDA reproduces it. CPU results retained as a separately labelled additional check.',initial_cpu_comparison_failed_fields=['random_logit','map_mean_absolute_difference','map_pearson_correlation'],initial_max_absolute_differences=[.234,.04834,.4482],primary_predictions_affected=False))
        cnn.cpu()
    else:
        check('Original CUDA randomization reproduction',False,'CUDA unavailable; CPU random weights are a separate randomization control',scope='not verified')
    table('randomization_original.csv',sanity)
    error=select_error_panel(csv('results/test_predictions.csv'),read('development/thresholds.json'),read('training/selection.json')['selected_traditional'])
    saved=csv('test_explanations/test_error_selection.csv')
    check('Deterministic structured test error selection',set(error.nodule_id)==set(saved.nodule_id) and len(error)==len(saved))
    table('structured_errors.csv',error)


def resource_audit():
    table('training_resources.csv',csv('resources/training_resources.csv'))
    samples=csv('resources/latency_samples.csv');print('Latency columns',list(samples),flush=True)
    # Stored long-form observations use one latency row per route and definition.
    value='seconds' if 'seconds' in samples else 'latency_seconds'
    rows=[]
    for (name,definition),g in samples.groupby(['model','definition']):
        a=g[value];rows.append(dict(model=name,definition=definition,median=a.median(),q25=a.quantile(.25),q75=a.quantile(.75),iqr=a.quantile(.75)-a.quantile(.25),samples=len(a)))
    fresh=table('latency_five_pass.csv',rows)
    compare('Five-pass resource summary',fresh,csv('resources/latency_summary.csv'),['model','definition'])
    component=csv('resources/component_samples.csv');print('Component columns',list(component),flush=True)
    rows=[]
    for c in component:
        if c.endswith('_seconds'):
            a=component[c];rows.append(dict(component=c,median=a.median(),q25=a.quantile(.25),q75=a.quantile(.75),iqr=a.quantile(.75)-a.quantile(.25),samples=len(a)))
    frame=table('latency_component_single_pass.csv',rows)
    saved=csv('resources/component_summary.csv')
    shared=frame[frame.component.isin(saved.component)]
    compare('Supplemental component summary',shared,saved[saved.component.isin(shared.component)],['component'])
    for n in ['measurement.json','component_measurement.json']:
        save('resources/'+n,read('resources/'+n))
    for n in ['source_decode_times.csv','component_source_decode_times.csv']:
        table(n,csv('resources/'+n))


def prototype_audit(config):
    from lidc_experiment.viewer import view_case
    from lidc_experiment.training import NoduleDataset
    import tempfile
    rows=[]
    for label,key in [('missing identifier',None),('unknown identifier','not-a-valid-prepared-case')]:
        try:
            view_case(SRC,config,key,OUT/'prototype/invalid.html');passed=False;detail='unexpected accepted'
        except ValueError as e:passed=True;detail=str(e)
        rows.append(dict(test=label,passed=passed,detail=detail))
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        for label,image in [('wrong dimensions',np.zeros((1,64,64),np.float32)),('nonfinite',np.full((2,64,64),np.nan,np.float32)),('empty mask',np.zeros((2,64,64),np.float32))]:
            p=Path(temp)/'bad.npz';np.savez(p,image=image)
            try:NoduleDataset(pd.DataFrame([dict(npz_path=str(p),label=0)]));passed=False;detail='unexpected accepted'
            except ValueError as e:passed=True;detail=str(e)
            rows.append(dict(test=label,passed=passed,detail=detail))
    text=(SRC/'prepared_case.html').read_text()
    for phrase in ['Research classification of radiologist-rated suspicion','no clinical validation','does not perform full-scan nodule detection','Supplied expert contours']:
        rows.append(dict(test='research-use label: '+phrase,passed=phrase in text,detail='existing static HTML source inspected'))
    key=csv('cohort.csv').query("partition == 'validation'").sort_values('nodule_id').nodule_id.iloc[0]
    file=view_case(SRC,config,key,OUT/'prototype/prepared_case_verified.html')
    rows.append(dict(test='generate prepared validation case',passed=Path(file).is_file(),detail=file))
    for r in rows:check('Prototype '+r['test'],r['passed'],r['detail'])
    table('prototype_acceptance.csv',rows)


def finish():
    original=json.loads((OUT/'preservation/original_hashes.json').read_text())
    check('Original artifact bytes unchanged at audit end',all((ROOT/p).is_file() and sha(ROOT/p)==d['sha256'] for p,d in original['files'].items()),f'{len(original["files"])} artifact files')
    check('Original implementation unchanged',all(sha(ROOT/p)==digest for p,digest in original['source_sha256'].items()))
    docs=json.loads((OUT/'documents/source_hashes.json').read_text())
    check('Dissertation and source documents unchanged',all(sha(p)==r['sha256'] for p,r in docs.items()))
    table('verification_checks.csv',CHECKS)
    table('numerical_discrepancies.csv',pd.DataFrame(DISCREPANCIES,columns=['item','detail','impact']))
    save('audit_summary.json',dict(execution_complete=COMPLETED,total_checks=len(CHECKS),passed=sum(x['passed'] for x in CHECKS),failed=sum(not x['passed'] for x in CHECKS),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),raw_ct_available=Path(read('run.json')['config']['data_root']).is_dir(),post_test_inspection_analysis=True,no_training_performed=True))
    print((OUT/'audit_summary.json').read_text(),flush=True)


def main():
    global COMPLETED
    COMPLETED=False
    CHECKS.clear()
    DISCREPANCIES.clear()
    OUT.mkdir(exist_ok=True,parents=True)
    try:
        print('Preservation',flush=True);preservation()
        print('Cohort',flush=True);cohort,features=cohort_audit()
        print('Training provenance and reload',flush=True);models,config=training_audit(cohort,features)
        print('Independent metrics and 2000 bootstrap draws',flush=True);predictive_audit(cohort)
        print('Secondary and explanations',flush=True);secondary_audit(cohort,models,config)
        print('Resources',flush=True);resource_audit()
        print('Prototype',flush=True);prototype_audit(config)
        COMPLETED=True
    finally:
        finish()
    if any(not result['passed'] for result in CHECKS):
        raise SystemExit(1)


if __name__=='__main__':main()
