"""Re-extract features, mask predictions and permutation importance; no refits."""
from pathlib import Path
import contextlib
import json
import logging
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lidc_experiment.features import build_extractor, extract_case
from lidc_experiment.training import load_model,predict_model
from lidc_experiment.explanations import _feature_explanations

SRC=ROOT/'artifacts/chapter3'
OUT=ROOT/'results_evidence/2026-11-12'


def main():
    config=json.loads((SRC/'run.json').read_text())['config']
    config['runtime']['device']='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    import SimpleITK as sitk
    import radiomics
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    radiomics.setVerbosity(logging.ERROR)
    extractor,_=build_extractor(config)
    prepared=pd.read_csv(SRC/'data/all_nodules.csv');features=pd.read_csv(SRC/'features/radiomics_features.csv').set_index('nodule_id')
    records=[]
    for i,row in enumerate(prepared.itertuples()):
        fresh,_=extract_case(row.npz_path,extractor)
        values=features.loc[row.nodule_id,list(fresh)].to_numpy(float);a=np.array(list(fresh.values()))
        records.append(dict(nodule_id=row.nodule_id,feature_count=len(fresh),max_absolute_difference=float(np.nanmax(np.abs(values-a))),matches=bool(np.allclose(a,values,rtol=1e-8,atol=1e-8,equal_nan=True))))
        if (i+1)%200==0:print(f'Features re-extracted {i+1}/{len(prepared)}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'tables/feature_reextraction_audit.csv',index=False)
    models={n:load_model(SRC/'training/models'/(n+('.pt' if n=='cnn' else '.joblib'))) for n in ['logistic_regression','svm','random_forest','cnn']}
    cohort=pd.read_csv(SRC/'cohort.csv').set_index('nodule_id');masks=pd.read_csv(SRC/'secondary/mask_sensitivity.csv');rows=[];feature_rows=[]
    for (key,variant,reader),g in masks.groupby(['nodule_id','variant','reader_index'],dropna=False):
        suffix='all' if pd.isna(reader) else str(int(reader))
        stem=f'{key}_{variant}_{suffix}';path=SRC/'secondary/masks'/(stem+'.npz')
        feature,_=extract_case(path,extractor);replacement=pd.DataFrame([dict(nodule_id=key,**feature)])
        saved_features=pd.read_csv(SRC/'secondary/masks'/(stem+'_features.csv')).set_index('feature')
        for f,v in feature.items():
            feature_rows.append(dict(nodule_id=key,variant=variant,reader_index=reader,feature=f,regenerated_value=v,saved_value=saved_features.loc[f,'alternative_value'],matches=bool(np.isclose(v,saved_features.loc[f,'alternative_value'],rtol=1e-8,atol=1e-8))))
        case=cohort.loc[[key]].reset_index().copy();case['npz_path']=str(path)
        with np.load(path,allow_pickle=False) as a:metadata=json.loads(str(a['metadata'].item()))
        for row in g.itertuples():
            predicted=predict_model(models[row.model],case,replacement,config).score.iloc[0]
            rows.append(dict(nodule_id=key,variant=variant,reader_index=reader,model=row.model,regenerated_score=predicted,saved_score=row.alternative_score,absolute_difference=abs(predicted-row.alternative_score),matches=bool(np.isclose(predicted,row.alternative_score,rtol=1e-6,atol=1e-7)),slice_matches=metadata['selected_slice_index']==row.alternative_slice))
    pd.DataFrame(rows).to_csv(OUT/'tables/mask_prediction_reproduction.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(OUT/'tables/mask_feature_reproduction.csv',index=False)
    out=OUT/'explanations_recomputed';out.mkdir(exist_ok=True)
    _feature_explanations(models,cohort.reset_index().query("partition == 'validation'"),features.reset_index(),config,out)
    fresh=pd.read_csv(out/'permutation_importance.csv').set_index(['model','feature']).sort_index()
    saved=pd.read_csv(SRC/'explanations/permutation_importance.csv').set_index(['model','feature']).sort_index()
    cols=['baseline_auc','mean_auc_decrease','std_auc_decrease']
    perm=bool(np.allclose(fresh[cols],saved.loc[fresh.index,cols],rtol=0,atol=1e-12,equal_nan=True))
    result=dict(prepared_cases=len(records),all_original_features_match=all(r['matches'] for r in records),original_feature_tolerances={'rtol':1e-8,'atol':1e-8},alternative_prediction_rows=len(rows),all_mask_predictions_match=all(r['matches'] and r['slice_matches'] for r in rows),all_mask_features_match=all(r['matches'] for r in feature_rows),permutation_rows=len(fresh),permutation_results_match=perm,no_training=True)
    (OUT/'representation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
