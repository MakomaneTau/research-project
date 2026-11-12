"""Assemble separate review documents from the additional verification outputs."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_evidence/2026-11-12'
SRC=ROOT/'artifacts/chapter3'


def load(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def frame(name):return pd.read_csv(OUT/'tables'/name)
def write(name,text):(OUT/name).write_text(text.strip()+'\n',encoding='utf8')
def md(data,columns=None):
    f=data.copy()
    if columns:f=f[columns]
    def fmt(x):
        if pd.isna(x):return 'undefined'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])


def main():
    for source,target in [('data/source_panels','preprocessing'),('explanations/development_panels','development_explanations'),('test_explanations/test_error_panels','test_errors')]:
        destination=OUT/'figures'/target;destination.mkdir(exist_ok=True,parents=True)
        for p in (SRC/source).glob('*.png'):shutil.copyfile(p,destination/p.name)
    shutil.copyfile(SRC/'secondary/learning_curves.png',OUT/'figures/learning_curves.png')
    median=pd.read_csv(SRC/'secondary/median_label_changes.csv')
    summary=frame('median_cohort_changes.csv').set_index('partition')
    for name,g in median.groupby('partition'):
        summary.loc[name,'median_low']=int((g.label==0).sum())
        summary.loc[name,'median_high']=int((g.label==1).sum())
    summary.reset_index().to_csv(OUT/'tables/median_cohort_changes.csv',index=False)
    acquired=pd.read_csv(OUT/'tables/acquired_series_inventory.csv')
    flow=frame('cohort_flow.csv')
    flow=flow[flow.stage!='acquired DICOM directories (all modalities)']
    flow=pd.DataFrame([dict(stage='acquired DICOM directories (all modalities)',patients=acquired.patient_id.nunique(),series=acquired.series_uid.nunique(),nodules=None,unit_note='current DICOM storage; CR/DX are not annotated CT candidates')]+flow.to_dict('records'))
    flow.to_csv(OUT/'tables/cohort_flow.csv',index=False)
    audit=load('audit_summary.json');pres=load('preservation/original_hashes.json')
    cluster=load('clustering_audit.json') if (OUT/'clustering_audit.json').exists() else None
    source=load('source_audit.json') if (OUT/'source_audit.json').exists() else None
    representation=load('representation_audit.json') if (OUT/'representation_audit.json').exists() else None
    stats=load('bootstrap_audit.json');metrics=frame('metrics.csv')
    categories=[
        ('Repository and scientific source','verified','HEAD 92c854a9a69828624bc13a2999b4b8529e31d8de; original tracked files unchanged; fitted training implementation hash matches current source.'),
        ('Original local artifacts','verified',f'{pres["file_count"]} files archived with independently checked SHA-256; 608/608 frozen and 524/524 completion-delivery hashes match.'),
        ('Executed notebook','verified','Saved cells inspected with no recorded errors; metrics cross-checked against CSVs; notebook was not re-executed.'),
        ('Cohort, exclusions, splits and folds','verified','Counts, rating vectors and grouping keys reconstructed; patient, series, SOP and fold overlap zero.'),
        ('Candidate selection, final models, early histories','verified','144 successful fold records; 48 candidates; final and fold fit provenance; transformations and training-only SVM calibration checked; final predictions reloaded.'),
        ('Validation thresholds and test predictions','verified','304 validation and 360 test prediction rows, four common models; thresholds and metrics recalculated from preserved predictions.'),
        ('Statistical outputs','verified','Independent 2000-valid paired patient-bootstrap with seed 42 and repeated-patient multiplicities reproduces intervals.'),
        ('Annotation grouping equivalence','verified' if cluster and cluster['all_native_custom_groups_match'] and cluster['audited_annotation_ids_match'] else 'recorded but unverified','See clustering_audit.json and tables/clustering_equivalence.csv; equivalence to installed algorithm is distinct from expert anatomical adjudication.'),
        ('Original source CT and prepared-case reconstruction','verified' if source and source['source_sop_sha256_mismatches']==0 and source['all_qa_arrays_identical'] else 'recorded but unverified','See source_audit.json; full retained SOP checksums for prepared series and nine deterministic development reconstructions. Raw drive was reconnected during review.'),
        ('Feature extraction and alternate masks','verified' if representation and representation['all_original_features_match'] and representation['all_mask_predictions_match'] else 'recorded but unverified','See representation_audit.json; all prepared features and alternate-mask scores re-extracted without refitting.'),
        ('Explanations and 29 development robustness fits','verified','Saved predictions, training boundaries, LR coefficients, CUDA Grad-CAM control and deterministic test error selection checked. Claims remain restricted.'),
        ('Resource samples','verified','Five-pass and supplemental one-pass summaries recalculated; original measurement conditions are recorded, not independently rerun.'),
        ('Prototype programmatic checks','verified','Valid prepared-case HTML generated, model reloads agree, invalid identifiers/tensors rejected, research labels present.'),
        ('Prototype browser display/usability','missing','Browser access host unavailable; native browser launch approval timed out. Separate hash-displaying interface copy supplied. Programmatic checks do not establish observed browser layout or user acceptance.'),
        ('Historical retrieval route/date/version, usage record','missing','Current acquired files can be inventoried; no historical download/usage record was found. Current hashes do not establish retrieval date or permission to redistribute.'),
        ('Issued ethics approval and practical-scope determination','missing','Located Ethics Form.pdf is applicant/supervisor-signed, with committee decision/reference/chair fields blank. It cannot establish issued approval or scope coverage.'),
        ('Draft versus handover version description','inconsistent','Current supplied Revised Draft.pdf has 80 pages; handover describes 68. Current PDF is the reviewed methodology. Neither document was edited.'),
        ('Ambiguous anatomical group adjudication','recorded but unverified','Eight groups remain quarantined; saved numeric review explicitly does not claim expert/manual anatomical adjudication.'),
    ]
    inv=pd.DataFrame(categories,columns=['item','status','verification_boundary'])
    inv.to_csv(OUT/'tables/evidence_category_inventory.csv',index=False)
    write('EVIDENCE_INVENTORY.md',f'''# Evidence inventory - 12 November 2026

{md(inv)}

The four inventory states apply to the stated verification boundary. Every original file is separately listed in [evidence_inventory.csv](tables/evidence_inventory.csv), with its size, SHA-256 and archive verification. That file-level verification does not certify every scientific interpretation. Read this category inventory with [verification_checks.csv](tables/verification_checks.csv) and the source/representation/clustering audit JSONs.

The original directory remains `artifacts/chapter3`; it was not overwritten. Its full preservation copy is [chapter3_original.zip](preservation/chapter3_original.zip), with [original_hashes.json](preservation/original_hashes.json). This local preservation copy includes patient-linked research records and supplied contours. Public redistribution has not been authorized by this review.

Repository state at initial inspection was clean `main` at `{audit['commit']}`. This equals the earlier review commit. The historical run's environment identifies `0bad890` plus uncommitted implementation, rather than claiming the later documentation commit existed at training time. Current training.py bytes match fitted-model metadata; the later commit records those implementation bytes. No commit or push was made for this review.
''')
    source_text=(f'''Current storage inventory: {source['raw_patient_folders']} patient-named directories; {source['acquired_series_directories']} DICOM series directories. Modalities: {source['modality_counts']}. All {source['database_series']} annotation-database CT paths are accessible. Directory names without a corresponding database patient are {source['folders_without_database_patient']}; directory existence alone is not an additional analysed patient. The database cohort comprises {source['database_patients']} patient identities.

For every one of the {source['prepared_series_checked']} prepared source series, retained source SOP identity sets were checked against preserved metadata. SHA-256 mismatches: {source['source_sop_sha256_mismatches']}. Bytes read: {source['prepared_source_bytes_checked']:,}. Size/mtime/annotation fingerprints match: {source['source_fingerprints_match']}. All nine deterministic development QA cases were reconstructed from raw CT; exact HU, mask, two-channel image and slice identity agreement: {source['all_qa_arrays_identical']}. Geometry exclusions reproduced: {source['geometry_exclusions_reproduced']}. This verifies retained source bytes and the sampled transformation path; it is not a fresh expert review of every contour or a historical acquisition record.'''
                 if source else 'Full source audit is still running or unavailable. Until source_audit.json is complete, raw-source verification is not ready.')
    cluster_text=(f'''The custom and installed pylidc grouping state machines produce identical memberships for all {cluster['eligible_spacing_series']} database series satisfying the spacing screen when supplied the cached exact KD-tree minimum distances. Reconstructed reader/diameter/series selection yields {cluster['reconstructed_nodules']} candidates and identical retained annotation-ID sets: {cluster['audited_annotation_ids_match']}. Exact tree-minimum distances were independently compared with chunked cdist for {cluster['distance_sample_pairs']} deterministic pairs, including ambiguous-series pairs; maximum difference {cluster['maximum_distance_difference']:.3g}. The full native default cdist matrix was not materialized for every pair in every scan: the exhaustive test concerns grouping/selection with exact tree minima, while the cdist equivalence check is sampled. The algorithms compute the same Euclidean minimum on the same coordinates, but equivalence to pylidc is not expert anatomical truth.

Equivalence has two qualifications. Installed pylidc 0.2.3 passes contour index coordinates to its distance calculation; the custom implementation preserves those coordinates. The variable name tolerance_mm must not be used as proof of a patient-space Euclidean metric. Also, the custom docstring says pylidc raises for unresolved groups, whereas this installed version breaks and returns them. This is a documentation error, not a detected membership difference. The custom deterministic secondary sort changes display order at tied centroids, not group identity or hash-derived nodule keys.'''
                  if cluster else 'Clustering audit is not complete; no equivalence claim is yet made.')
    model=frame('model_selection_and_training.csv')
    write('IMPLEMENTATION_REVIEW.md',f'''# Separate implementation review and results evidence

## 1. Evidence inventory and provenance

Begin with [EVIDENCE_INVENTORY.md](EVIDENCE_INVENTORY.md). The experiment is recoverable, and the current repository is still the earlier reviewed commit. All original scientific files remain unchanged. The additional verification is dated 12 November 2026 and was performed after test results were already known. No experiment model was retrained, no candidate was selected using test performance, and no original output was replaced. Existing tests do fit small synthetic models; those are software tests, not reruns of the LIDC experiment.

The supplied revised PDF, updated handover, rubric and template were read. Source hashes are in `documents/source_hashes.json`. The rubric's page 2 assigns Chapter 4 twenty marks for presentation, analysis and interpretation using appropriate techniques. Template page 8 places detailed data/results in Chapter 4, literature comparison in Chapter 5, and overall conclusions/recommendations in Chapter 6. This package is a review and supporting evidence, not dissertation chapter text.

The 80-page revised PDF supersedes the handover's 68-page description for this review. It specifies eight objectives, five questions and the matched 2D experiment. The handover's statement that practical completion was unconfirmed is a document-history statement; the local retained artifacts now provide direct evidence of implementation.

## 2. Acquisition, cohort construction and independence

{source_text}

{md(frame('cohort_flow.csv'))}

These rows describe nested retained sets; excluded intermediate cases are a subset of prepared cases, not another incoming acquisition. The original collection is neither 1,594 nodules nor 545 nodules. The 1,049 intermediate-rated nodules are 65.81% of 1,594 prepared nodules; the binary task retains 34.19%. Patients may contribute both intermediate and extreme-rated nodules, so the intermediate-patient count cannot simply be subtracted from 689 to obtain 368.

Series selection first screens actual annotated-series metadata: positive through-plane spacing no greater than 2.5 mm and at least one group with two to four annotation instances. Among candidates for each patient, smallest spacing wins, with lexicographic Series Instance UID resolving ties. Reader/diameter rules are then applied to all groups of that selected series. Reader counts are annotation-instance counts under pylidc grouping, not independently verified reader identities. Ratings are integers on 1-5 and their arithmetic mean defines low <=2, high >=4, intermediate otherwise. All 1,594 saved vectors, means, medians, reader counts and primary labels were checked. Diameter uses the mean available annotation diameter, with a minimum of 3 mm.

{md(frame('exclusions_by_unit.csv'))}

The 326 series exclusion events comprise 118 spacing exclusions, 207 without an eligible multi-reader group and one additional eligible series. Thus 1,018 - 326 = 692 selected series. Within selected series, 587 insufficient-reader groups and eight ambiguous groups are exclusion events; they are not 595 additional patients or an enumeration of all nodules in discarded series. There are 1,598 eligible candidate nodules in 692 patients; four geometry failures remove four nodules across three patients, leaving 1,594 nodules in 689 patients. No feature-extraction failures remove further cases. The 1,049 intermediate exclusions then leave 545 nodules in 368 patients.

{md(frame('class_distribution.csv'))}

The final test has 90 nodule predictions from 56 independent patient groups, with 32 high-suspicion and 58 low-suspicion nodules. Nodule predictions are not 90 independent patients. Test patients contribute one to five nodules (median one); training and validation ranges are one to nine and one to four, respectively. [patient_nodule_contributions.csv](tables/patient_nodule_contributions.csv) lists every contribution. The patient stratum is whether at least one eligible nodule is high-suspicion. Reapplying seed-42 patient splitting and grouped folds reproduces the stored assignment. [leakage_audit.csv](tables/leakage_audit.csv) reports zero overlap for patient, nodule, source-series and source-image identifiers across partitions, and zero patient overlap across training folds. Source-image checking uses entire selected-series SOP lists, a stronger condition than checking only the chosen slice.

{cluster_text}

Eight unresolved spatial groups in selected series remain excluded. The clustering audit finds thirteen unresolved groups across all 900 spacing-eligible series; only eight are selected-series exclusion events, so these denominators must not be conflated. The saved `flagged_case_review.json` documents annotation identities and native-grid spatial checks; it explicitly disclaims expert adjudication. Quarantine preserves an analysable unambiguous subset but does not resolve those groups or establish an exhaustive anatomical census. Any later adjudication that changes eligibility or labels must remain a separate amended analysis, with the already-exposed test boundary disclosed.

Geometry follows physical DICOM positions/orientation, HU rescale slope/intercept and regular-grid checks. Reader masks are sampled with nearest-neighbour interpolation onto the common 1 mm reference before majority aggregation. Intensities are sampled linearly once on the selected maximum-consensus-area slice. Ties use centroid distance and then the lower slice index. Local grid-aligned resampling is an implementation optimization of the same affine mapping; synthetic tests compare it with a full reference. Cropping rejects truncated selected masks; padding is fixed air HU with zero mask. All 1,594 retained arrays satisfy the cached 64x64 HU/mask and 2x64x64 masked-input contract. The complete selected 2D mask is the crop criterion; this is not a claim that a 3D lesion is represented in full.

Both pathways use this exact selected HU slice and mask; radiomics sees the masked 2D region, and the CNN receives masked normalized intensity plus the binary mask. Reader ratings, identifiers and neighbouring anatomy are excluded as predictors. Retained features are exclusively allowlisted original-image families. Feature settings specify shape2D, firstorder, glcm, glrlm and glszm, bin width 25 HU, force2D=true and force2Ddimension=0. [representation_audit.json](representation_audit.json) gives full re-extraction checks; [prepared_array_audit.csv](tables/prepared_array_audit.csv) covers all cached cases. Physical correctness is supported by source hashes, synthetic tests and sampled raw reconstructions, not by array shape alone.

## 3. Model selection and training verification

There are twelve candidates per model and three common grouped training folds: 48 candidates and 144 successful fold records. No primary fold is missing or failed in the retained corrected search. [complete_candidates.csv](tables/complete_candidates.csv) preserves every candidate's settings and three fold AUCs. [cv_provenance_audit.csv](tables/cv_provenance_audit.csv) checks held-out IDs, fit IDs and recomputed AUCs for every fold. Median imputation, all-missing/constant filtering, alphabetical absolute-correlation pruning above 0.90, and LR/SVM scaling are learned on each permissible training subset. There are 83 extracted original features; final traditional pipelines each retain 41, with fold-specific dimensions reported separately. Deterministic feature extraction before partition assignment does not estimate any cohort-fitted feature-processing statistic; learned transforms remain fold-contained.

{md(model,['model','selected_candidate','settings','fold_aucs','mean_cv_auc','retained_features','fold_best_epochs','final_epochs','fit_wall_seconds','search_wall_seconds','device','precision'])}

Selection maximizes mean training grouped-fold AUC, with tolerance 0.0001 and written grid-order tie resolution. The traditional ordering is LR, SVM, RF for a model-level tie. RF is the selected traditional comparator by training CV, independent of its eventual test AUC. Each final model was fitted on exactly the 379 training nodules from 257 patients. The training.py hash stored in final metadata matches the current implementation. Training IDs and selected settings match each final checkpoint.

SVM's primary score is its signed decision value. Its probability is a learned sigmoid of pooled out-of-fold training scores. All calibration fold complements and held patient groups were verified disjoint; OOF records cover the training set only. A calibration optimizer precision failure is documented in `protocol_corrections.json` at 2026-10-31 23:38:51 UTC, before test evaluation: analytic-gradient BFGS replaced the unsuccessful optimizer while keeping the objective, grids, cohort and partitions. The earlier search is preserved in `training_before_calibration_fix/`; the full primary search was repeated. Do not combine archived incomplete-search timing with the corrected primary-search cost without explicitly defining that additional cost.

The CNN's chosen fold best epochs are 1, 9 and 1. Their median is one, exactly the specified final-duration rule. All 36 inner histories, their patient boundaries and stopping records were checked; [cnn_inner_histories.csv](tables/cnn_inner_histories.csv) retains every epoch. Final training has one loss entry, 0.7018616800, uses unweighted BCE, seed 42, augmentation and float32 CUDA, and records 2.546146 s final-fit time. The architecture has 25,553 trainable parameters; 379 observations at batch size 32 produce twelve optimizer batches in that final epoch. The full CNN search records 51.544080 s, including inner early stopping, outer refits and scoring. These are measurements for a compact model on a small cached cohort and documented GPU; they do not prove general training efficiency.

Test CNN probabilities are distinct for all 90 nodules but compressed to approximately 0.512246-0.522401; validation spans 0.512376-0.526744. This permits high rank discrimination while producing poor absolute probability calibration. The recorded one-epoch rule and history explain the short final fit, and reloading reproduces the outputs. They do not prove that extra epochs would help, that the software is defective, or that undertraining is the sole cause. Testing alternative training durations now would be exploratory model development requiring a new independent evaluation for confirmatory claims. No such redesign was performed.

## 4. Predictive results and patient-cluster uncertainty

All metrics below were recomputed from the preserved predictions, with high radiologist-rated malignancy suspicion as positive. The endpoint is a reader-derived imaging proxy. It is not pathology-confirmed cancer, a cancer diagnosis, or a patient-level clinical risk estimate.

{md(metrics,['model','roc_auc','average_precision','accuracy','balanced_accuracy','sensitivity','specificity','precision','f1','brier_score','tp','fn','tn','fp'])}

All saved test metrics, patient-weighted summaries, probability-0.50 results, subgroup summaries and calibration bins agree within absolute tolerance 1e-12. Final-model predictions agree at rtol=1e-6 and atol=1e-7; exact maximum differences are in [reload_predictions_audit.csv](tables/reload_predictions_audit.csv). Model reload is an integrity check and did not replace the primary saved prediction files.

The independently implemented paired bootstrap sorts the common nodule IDs, samples all 56 patient groups with replacement, carries all their nodules and preserves repeated draw multiplicities. It uses seed 42, 2,000 valid replicates and a 20,000-attempt cap. It obtained {stats['valid']} valid draws in {stats['attempts']} attempts with {stats['rejected_single_class']} single-class rejections. [bootstrap_patient_draws.npz](tables/bootstrap_patient_draws.npz) stores every patient draw; [bootstrap_replicates.csv](tables/bootstrap_replicates.csv) stores every model/metric replicate.

{md(frame('primary_comparison.csv'))}

The primary AUC difference is CNN minus CV-selected RF: -0.0255926724, 95% interval [-0.0590945363, -0.0024786246]. It favours the selected RF for this cohort and conditional endpoint. H1 is not supported. Weak CNN performance or an unsupported hypothesis is a legitimate experimental result. Individual intervals are in [metric_intervals.csv](tables/metric_intervals.csv). These are conditional on fixed models, settings and thresholds and quantify test-patient sampling; they exclude full selection/training uncertainty. Other comparisons remain secondary without a collection of superiority tests.

Patient weighting assigns each patient's nodules weights summing to one. It retains nodule outcomes while changing the estimand; it is not patient-level disease classification. Its AUCs are CNN 0.940831, LR 0.970031, RF 0.971998 and SVM 0.963235. Full weighted confusion totals are fractional and should not be printed as literal patient counts. The implementation's fixed subgroup bins use diameter <=6, >6-10 and >10 mm, and rating-range 0-1 versus >=2, implementing the planned diameter/disagreement descriptions. The <=6 mm test group contains 19 nodules and zero high-suspicion cases, so its AUC is undefined. The middle group has only three high-suspicion nodules among 38, while >10 mm has 29 among 33. The CNN's middle-group AUC is 0.390476, a sparse descriptive result that must not be hidden by its high overall AUC. Strong size/reference association and spectrum restriction limit broad discrimination claims. Denominators and undefined single-class metrics are retained in [subgroup_metrics.csv](tables/subgroup_metrics.csv); sparse categories do not support fairness or superiority claims.

## 5. Threshold behaviour and calibration

Validation thresholds were independently recovered as the highest observed native score achieving sensitivity >=0.90. With 34 high-suspicion validation nodules, at least 31/34 positives must be detected. Decisions use >=, retaining ties. Thresholds remain CNN 0.5145624229, LR 0.6916338315, RF 0.642 and SVM 0.2562173339 in its native decision scale. None is retuned on the test set.

Every test sensitivity at the frozen thresholds is below 0.90. This reflects uncertainty and changed patients, not an implementation rule guaranteeing test sensitivity. CNN detects 24/32 high-suspicion nodules and misses eight. At probability 0.50 the CNN marks all 90 nodules high-suspicion (TP=32, FP=58, TN=0, FN=0), illustrating why ranking and probability interpretation must be separated. LR, RF and calibrated SVM at 0.50 each yield TP=28, FN=4, TN=56, FP=2. These are secondary fixed-probability results; SVM 0.50 is applied to calibrated probability, never to the native decision value.

Brier scores are CNN 0.252447, LR 0.057953, RF 0.055249 and SVM 0.052811. Five equal-frequency bins retain 18 test nodules each, but shared patients and small bin sizes limit calibration precision. The CNN plot's nearly vertical range is consistent with its compressed probabilities. Numerical discrimination must not be described as well-calibrated cancer risk. ROC, PR, four confusion matrices and calibration plots are in [figures/predictive](figures/predictive/).

## 6. Explanations and robustness

LR coefficients were checked directly against the frozen model with aligned feature names and standardized directions. For example, MinorAxisLength has coefficient +0.219998, RunEntropy +0.217902 and InverseVariance +0.203336. These are model log-odds associations per training standard deviation, not biological causes. All signed coefficients are supplied. Twenty-repeat validation permutation results are separately reproduced without refitting; grouped dependence and correlated predictors limit causal or stability claims.

Twelve deterministic development Grad-CAM panels and all 21 shared structured test-error panels are retained. Grad-CAM targets the high-suspicion logit at the last convolution. The saved fully randomized-network control has nonconstant trained and randomized maps in 12/12 cases, but trained-versus-randomized correlations remain approximately 0.6365-0.8439. Thus maps remain substantially similar after randomisation. Changed logits/maps demonstrate some parameter dependence, but retained spatial similarity prevents a strong claim that salient regions uniquely reflect learned disease reasoning. Neither localization validity, clinician trust nor interpretability superiority is established.

The initial extra CPU repeat produced different randomized weights because CPU and CUDA initialization use different random streams. Repeating reset_parameters on the original CUDA device reproduces the recorded control. CPU outputs are retained separately in `randomization_recomputed_cpu.csv`; this device-specific check did not modify primary predictions. It illustrates why a numerical seed must be accompanied by device and implementation information.

All 29 original secondary fits use development patients only. Their recorded predictions and training boundaries were checked; no missing prespecified analysis required fresh training. These comprise 16 learning-curve fits, four class-weight toggles, one no-augmentation CNN, four RF/CNN seed repeats and four median-label fits. Seeds 17 and 73 apply to the stochastic RF/CNN pathways under the protocol. The primary seed remains 42. The original secondary stage preceded freeze/test evaluation; today's verification of it is after test inspection.

{md(frame('median_cohort_changes.csv'))}

Median labels add 57 training and six validation nodules, with no retained low-to-high/high-to-low flips and no losses from the mean-binary development set. There are 436 median-labelled training nodules and 82 validation nodules, total 518. Patients outside the primary development allocation remain excluded; this is not a median census of all 1,594 cases. Median validation AUCs assess a changed task/cohort and must not be treated as a paired primary-test comparison.

{md(frame('mask_robustness_summary.csv'))}

The 29 deterministic validation cases produce 29 strict-majority and 87 individual-reader alternatives per model, 464 model-prediction rows. Compare changes within each score scale. In particular, SVM's decision units cannot be directly ranked against probability changes, and the CNN's small changes are constrained by its already compressed scale. Thirty-four individual-mask and eight strict-mask variants change the selected slice; those counts repeat across models because preprocessing is shared. Alternate feature values and prediction reproduction are supplied. Neither visually stable maps nor small native score changes alone establish clinical robustness.

The no-augmentation CNN's validation AUC equals the primary development value at the recorded precision (0.976891). Seed-17/73 CNN validation AUCs are 0.980392 and 0.974090; RF is 0.990196 for both. Learning-curve differences are descriptive on one shared validation set, not proof of a monotonic sample-size effect. [robustness_development.csv](tables/robustness_development.csv) contains all settings, sample sizes and results. No best seed, label rule, mask rule or weighting alternative replaces the primary models.

## 7. Resource measurements and prototype

Original hardware records identify Windows build 26300, AMD64 Family 25 Model 68 Stepping 1, 12 logical CPUs, 25,042,907,136 bytes RAM, and NVIDIA GeForce RTX 2050 with 4,294,443,008 bytes GPU memory. A consumer CPU marketing name and physical-core count were not recorded; do not invent them. Python is 3.11.9; key versions include numpy 1.26.4, scipy 1.13.1, sklearn 1.6.1, torch 2.11.0+cu128, SimpleITK 2.5.3, pydicom 3.0.1, pylidc 0.2.3 and PyRadiomics 3.1.0. Environment-level deterministic flags were captured before training; per-fit metadata records deterministic training=true. These snapshots describe different times and are not evidence that training disabled determinism.

CPU fits use float64; CNN fitting uses float32 CUDA without automatic mixed precision. CPU inference is batch one, one CPU thread and one SimpleITK thread, with explicit threadpool limits. RF has 500 trees and one configured worker. LR uses lbfgs with maximum 5,000 iterations; SVM fitting has probability mode disabled. The training code sets torch threads and RF workers but does not enforce/capture an equivalent BLAS threadpool limit for LR/SVM fits; exact training BLAS threading remains unverified and must not be inferred from the configured cpu_threads field. [training_resources.csv](tables/training_resources.csv) separates successful corrected search and final fitting, not total project/development time or cumulative abandoned-run cost. Final-fit clocks exclude pre-fit fingerprint computation and post-fit serialization. RSS is absolute whole-process memory sampled every 20 ms, including already imported state, not isolated model memory; brief peaks can be missed between samples. CUDA peak allocated memory excludes reserved/context/driver memory. Saved file size excludes external source images and environment.

{md(frame('latency_five_pass.csv'))}

Latency units are seconds. Each route/definition has 450 samples, five passes over 90 test nodules. Model-only starts with an already transformed numeric vector/tensor and has ten explicit warm-up predictions per model; traditional feature transformation is outside its model-only clock. The stored decoded-CT route is a partial component sum: common preparation plus model prediction, fitted feature transformation for traditional models and uncached radiomic extraction for those models. Original tensor conversion and one-row dataframe assembly occur outside these clocks. There are no ten explicit full-path warm-ups for the original five-pass route; the metadata's unqualified warmups=10 describes the model-only loop. This is a protocol/measurement-boundary limitation. These values must be labelled partial route sums, not complete integrated pipeline latency. Disk decoding, annotation retrieval and human contouring are also excluded.

The supplemental component benchmark is a separate single pass of 90 nodules after ten full-path warm-ups. Common preparation median is 0.022209 s, radiomics extraction 0.016367 s, dataframe assembly 0.0009405 s, CNN tensor conversion 0.00005425 s and CNN forward processing 0.00166935 s. Route component sums are reported in [latency_component_single_pass.csv](tables/latency_component_single_pass.csv); they are sums of measured components, not an independently timed integrated trial. Do not merge them with the original five-pass samples. Separate original disk/CT integrity-decode/annotation observations cover 56 scans in five passes, 280 scan-level samples; costs are shared by nodules of the same scan and cannot be added once per nodule. Neither set includes network download or the original human contouring time.

Resource hypotheses are dimension-specific. The selected RF has less corrected search and final-fit time than the CNN on this hardware, but it has a larger serialized model and slower CPU model-only and decoded-CT route latency. Unequal CPU/GPU training prevents a pure algorithmic training-efficiency claim. There is no universal resource winner and no measured monetary saving.

Two post-test resource amendments preserve earlier manifests, patches and prediction hashes. The first initializes the existing pylidc compatibility adapter in the measurement runner; the second records separately saved components. A failed first component attempt records a source-read transport error after one nodule, followed by a full retry. It is not a scientific exclusion. These changes do not redesign or refit a model.

The prepared-case viewer accepts a saved annotated nodule ID and presents image, contour, native score, probability, threshold and decision. Programmatic acceptance checks reject missing/unknown IDs, wrong dimensions, nonfinite arrays and empty masks. Generated validation-case HTML is in [prototype/prepared_case_verified.html](prototype/prepared_case_verified.html). Labels state research suspicion classification, expert-contour dependency, absence of clinical validation and absence of full-scan detection. Final-model reloads match. The frozen interface refers to the artifact directory without showing exact model/configuration hashes. A separate interface-only correction, [prepared_case_with_provenance.html](prototype/prepared_case_with_provenance.html), now displays those exact hashes with an explicit post-test-amendment label; the original viewer and predictions are unchanged. Its machine-readable record is `prototype/interface_amendment.json`. Browser rendering was not verified because browser access was unavailable and the native-browser launch approval timed out; no clinician or end-user testing is claimed.

## 8. Discrepancies, readiness and remaining actions

See [DISCREPANCY_LOG.md](DISCREPANCY_LOG.md), [READINESS.md](READINESS.md), [CHAPTER4_STRUCTURE.md](CHAPTER4_STRUCTURE.md) and [TRACEABILITY.md](TRACEABILITY.md). The predictive result is reproducible. No label, split, selected-model or primary prediction defect has been found. The scientific interpretation remains conditional on this proxy-labelled, restricted, single-dataset 2D task and unresolved excluded clusters.

The package is evidence for an unfavourable CNN primary comparison, not evidence of clinical deployment readiness or universal ML superiority. Final dissertation release still needs the issued ethics/scope documentation, historical acquisition/usage provenance, disclosure of protocol differences and completion or explicit limitation of the prototype display/browser checks. Neither missing documentation nor weaker model performance should be silently converted into a new primary experiment on the exposed test cohort.
''')
    trace=[
        ('Objective 1','Critical literature synthesis','Revised Draft Chapter 2; evidence matrix; not rewritten here','Literature/source verification beyond this implementation review remains with dissertation author','Outside this package'),
        ('Objective 2','Traceable cohort','cohort_flow.csv; exclusions_by_unit.csv; source_audit.json; clustering_audit.json','Historical acquisition record and unresolved excluded anatomy','Ready with disclosed limitations'),
        ('Objective 3','Corresponding preprocessing inputs','prepared_array_audit.csv; feature_reextraction_audit.csv; raw_preprocessing_reconstruction.csv; figures/preprocessing','Sampled physical QA, no expert adjudication','Ready with disclosed limitations'),
        ('Objective 4','Four trained/tuned pipelines','complete_candidates.csv; cv_provenance_audit.csv; model_selection_and_training.csv; cnn_inner_histories.csv','One-epoch CNN follows the rule; no performance-driven replacement','Ready'),
        ('Objective 5 / RQ1 / H1','Common predictive comparison','metrics.csv; metric_intervals.csv; primary_comparison.csv; bootstrap draws; predictive figures','Conditional patient-sampling uncertainty and proxy/spectrum limits','Ready with disclosed limitations'),
        ('Objective 6 / RQ2 / H2a','Resource comparison','training_resources.csv; latency_five_pass.csv; latency_component_single_pass.csv; source decode tables','Different fitting hardware; RSS boundary; no whole-service costs','Ready with disclosed limitations'),
        ('Objective 6 / RQ3 / H2b','Forms and checks of explanation','logistic_coefficients.csv; permutation_importance.csv; randomization_original.csv; explanation panels','Spatial maps remain similar after randomisation; no numerical interpretability superiority','Ready with disclosed limitations'),
        ('Objective 7 / RQ4','Secondary sensitivity analyses','robustness_development.csv; median_cohort_changes.csv; mask_robustness_summary.csv','Development-only; changed median reference task; native score scales','Ready with disclosed limitations'),
        ('Objective 8 / RQ5','Auditable academic prototype and limits','prototype_acceptance.csv; prepared_case_verified.html; preservation manifest; READINESS.md','Exact IDs not visibly displayed; browser QA and formal practical-scope evidence absent','Not ready for unconditional acceptance'),
        ('Main research question','Controlled ML versus compact-CNN comparison','IMPLEMENTATION_REVIEW.md sections 2-7; metrics and resource tables','Applies to four selected pipelines, one cohort and shared supplied contours','Ready with disclosed limitations'),
    ]
    df=pd.DataFrame(trace,columns=['objective_or_question','purpose','evidence','remaining_gap_or_boundary','readiness']);df.to_csv(OUT/'tables/traceability.csv',index=False)
    write('TRACEABILITY.md','# Objective and question traceability\n\n'+md(df)+'\n\nPaths in the evidence column refer to `tables/`, audit JSONs or `figures/` as appropriate. Objective wording follows revised PDF pp. 13-14 and methodology Table 3.1. H2b is a structured explanatory proposition, not a numerical superiority test.')
    write('CHAPTER4_STRUCTURE.md','''# Proposed Chapter 4 structure: Results and Data Analysis

This is an evidence-led outline, not drafted dissertation prose. Chapter 4 carries 20 marks in the supplied rubric, page 2. Template page 8 requires detailed presentation of data, analysis and findings.

| Section | Tables and figures | Analysis and interpretation to support | Restrictions/gaps |
| --- | --- | --- | --- |
| 4.1 Experimental completion and cohort | Run/provenance table; acquisition-to-binary flow; exclusion units; class distributions | Separate source modalities/series, patients, nodules and exclusion events; identify corrected/frozen original run | Do not present prepared nodules as whole acquisition; link missing historical acquisition record |
| 4.2 Preprocessing and independence | Shared-input examples; raw reconstruction and overlap audit | Explain matched 2D information and all-zero overlap; clarify nodule prediction versus patient independence | Expert masks are required; unresolved clusters excluded; sampled visual review |
| 4.3 Model development and selection | Complete candidate appendix; selected settings/fold scores; retained features; CNN histories | Trace RF comparator to CV; explain one-epoch median duration without asserting a defect | No test-driven redesign; SVM correction disclosed |
| 4.4 Primary predictive comparison and uncertainty | Primary metric table; ROC/PR; four confusion matrices; individual and paired intervals | Report effect -0.02559 with interval [-0.05909,-0.00248]; H1 unsupported; denominators 90 nodules/56 patients | Conditional intervals; secondary comparisons descriptive |
| 4.5 Threshold behaviour and calibration | Frozen threshold table; probability-0.50 table; five-bin calibration/Brier; weighted/subgroup appendix | Explain validation sensitivity target versus observed test shortfall; CNN compressed probabilities | SVM decisions versus calibrated probability; sparse bins/groups |
| 4.6 Resource measurements | Search/final-fit/device/precision/memory/size table; five-pass latency; separate component/decode tables | Compare dimensions separately; spell out inputs and excluded decoding/human work | Mixed training devices, whole-process RSS, one-pass supplement |
| 4.7 Interpretability and robustness | Signed LR coefficients; permutation spread; trained/randomized Grad-CAM; error cases; learning curves; seeds; median/mask tables | Restrict claims when randomized maps remain similar; changed median cohort and within-model mask changes | No clinical reasoning, trust or causality claims; development-only alternatives |
| 4.8 Prototype testing and evidence limits | Acceptance matrix; research-use viewer; readiness/deviation register | Separate functional checks from visual/browser and clinical evidence; enumerate open documentation gates | No clinician testing, deployment claim, or unconditional ethics-scope assertion |

For each main table state the positive class, outcome reference, prediction unit, patient denominator, partition, weighting and threshold. Show point estimates with uncertainty, not only rankings. Captions should distinguish original analyses from November verification. Undefined metrics retain their denominators.

Appendix A: issued ethics/scope decision and acquisition/usage provenance (currently incomplete). Appendix B: manifests, exclusion units, fingerprints and source checks. Appendix C: all candidates, grouped-fit records, calibration OOF provenance and CNN histories. Appendix D: saved predictions, draw indices, replicate results and additional metric tables. Appendix E: all explanation and structured error panels. Appendix F: secondary fits/mask/feature changes. Appendix G: benchmark definitions, samples, amendments and acceptance logs. Appendix H: environment, commands, hashes and preservation archive.

Keep comparisons with prior literature for Chapter 5. Reserve overall conclusions, recommendations and future-work proposals for Chapter 6. No Chapters 5-6 were written and the revised dissertation was not edited. The rubric's appendices allocation is two marks; use clear indexes and relevant supporting evidence rather than an unlabelled artifact dump. Harvard referencing is specified on Template page 9; local experimental tables should cite their saved output and provenance, with literature references individually checked before publication.
''')
    write('DISCREPANCY_LOG.md','''# Discrepancy and protocol-deviation register

| ID | Observation | Timing and impact | Resolution / exact action |
| --- | --- | --- | --- |
| D01 | Supplied revised PDF is 80 pages; handover describes 68 | Document provenance discrepancy; no numerical effect | Use current PDF hash, identify reviewed version to supervisor, retain handover history |
| D02 | Original environment points to 0bad890 with dirty implementation; current HEAD is 92c854a | Run predates chronological documentation commits | Trace fitted training.py hash and frozen source hashes; do not claim later commit was training-time HEAD |
| D03 | SVM calibration optimizer failed before final test | Development correction; earlier search retained | protocol_corrections.json documents BFGS correction and full rerun; 144 corrected folds succeed |
| D04 | Custom grouping docstring says pylidc raises | Installed 0.2.3 returns unresolved groups instead; source comment inaccurate | No original source edit; actual group-membership equivalence checked; correct wording only in future documented source revision |
| D05 | Clustering tolerance named mm while upstream uses contour index matrices | Equivalence to upstream does not establish true patient-space distance | Disclose coordinate basis; any different physical-distance grouping changes cohort and must be separate |
| D06 | Eight ambiguous groups lack expert anatomical adjudication | Numeric identity/spatial review before freeze; groups quarantined | Retain exclusions and state incomplete adjudication; do not silently add/relabel cases |
| D07 | Local, aligned plane resampling replaces full-volume allocation | Deterministic implementation optimization, tested against full reference | Source reconstruction examples and synthetic physical-coordinate tests support equivalence |
| D08 | Two benchmark-only amendments after predictive evaluation | Compatibility initialization and separate component persistence; prior manifests retained | Report both amendments and preserved predictive hashes; keep five-pass and one-pass samples separate |
| D09 | First supplemental benchmark stopped after one case | Source transport/read failure; not a data-quality exclusion | Preserve attempt1.json and complete-retry output; never drop the case to make timing succeed |
| D10 | CPU Grad-CAM randomized model did not numerically match saved CUDA randomization | New November verification used a different seeded device RNG; no primary prediction difference | Repeating original CUDA initialization reproduces saved values; keep CPU control separately with its device |
| D11 | CNN probabilities are compressed, all exceed 0.50; one final epoch | Valid recorded rule/result with weak calibration | Report exact histories, range and Brier; no after-test primary replacement or unsupported defect diagnosis |
| D12 | Raw drive absent at first lookup, then reconnected by user | Temporary access issue, not missing historical experiment | Full current inventory and source-hash audit performed after reconnection |
| D13 | No historical acquisition/usage record found | Current inventory cannot establish retrieval date, release/version or redistribution terms | Supply original download/collection manifest, retrieval route/date/version record and usage/redistribution record |
| D14 | Located ethics form has blank committee decision/reference/chair fields | Applicant/supervisor declarations do not establish issued committee approval for changed practical scope | Supply actual approval letter/certificate and scope determination/amendment; do not infer refusal or approval |
| D15 | Viewer refers to artifact directory without exact model/configuration hashes on page | Original ID-display criterion only partially met | Separate prepared_case_with_provenance.html corrects display while preserving original viewer and prediction table; interface_amendment.json records exact hashes |
| D16 | Browser display/usability unobserved | Browser access host unavailable; native-browser launch approval timed out | Open generated prepared-case page and record visible labels, IDs, image/table layout and invalid-input workflow; no clinician-testing claim |
| D17 | Original five-pass route clock omits tensor/dataframe assembly and has no dedicated full-path warm-ups | Model-only has ten explicit warm-ups; metadata does not distinguish scope | Label original route values partial component sums; use separate one-pass supplement for complete component accounting; do not claim the original fulfilled a ten-warm-up complete-pipeline protocol |
| D18 | Training BLAS threadpool state was not enforced/captured for LR/SVM like CPU inference | A configured cpu_threads value is not proof of identical training thread use | Disclose recorded torch/RF controls and unknown BLAS state; avoid pure single-thread training-efficiency claims |

Independent saved-prediction metric and interval calculations agree within 1e-12. No unresolved numerical discrepancy affects the primary predictive result. Audit-script issues during development concerned the stored comparison JSON schema, the cross-device randomisation assumption and int32-versus-int64 DataFrame equality in the independent median-label check; the latter had identical values throughout. Final successful logs and the separate device note document their resolution. Historical scientific artifacts were never edited.
''')
    decisions=[('Primary saved predictive comparison','Ready with disclosed limitations','Paired predictions/thresholds/metrics/bootstrap reproduce; proxy labels, single dataset, restricted spectrum and fixed-model uncertainty'),('Cohort counts and partition independence','Ready with disclosed limitations','Manifests reconcile; no patient/series/SOP/fold overlap; ambiguous groups remain excluded'),('Model selection and training traceability','Ready','All selected models trace to training IDs, original grids, successful folds and final settings'),('Source CT identity and preprocessing','Ready with disclosed limitations' if source and source['source_sop_sha256_mismatches']==0 and source['all_qa_arrays_identical'] else 'Not ready','Full prepared-series source checks and sampled raw reconstruction; no exhaustive expert contour review'),('Feature and mask representation checks','Ready' if representation and representation['all_original_features_match'] and representation['all_mask_predictions_match'] else 'Not ready','Re-extraction and native-score checks; no changed primary models'),('Interpretability descriptions','Ready with disclosed limitations','Traceable features and nonconstant maps; randomized maps remain similar, so strong learned-localization or trust claims unsupported'),('Development robustness results','Ready with disclosed limitations','All 29 saved analyses use development data; changed median task and native score-scale boundaries'),('Resource tables','Ready with disclosed limitations','Saved sample summaries reproduce; device, process-memory, decoding and component boundaries disclosed'),('Prototype unconditional acceptance','Not ready','Functional checks pass, exact ID display and observed browser acceptance incomplete'),('Historical acquisition/usage and ethics-scope documentation','Not ready','Current files/form do not establish retrieval history, usage permissions or committee determination'),('Final dissertation result release','Not ready','Numerically verified results can be reviewed; unresolved documentation and prototype acceptance must be resolved or institutionally accepted as limitations')]
    ready=pd.DataFrame(decisions,columns=['result','classification','reason']);ready.to_csv(OUT/'tables/readiness.csv',index=False)
    write('READINESS.md','# Final readiness decision\n\n'+md(ready)+'''

The recorded CNN-minus-selected-RF AUC result is a valid, reproducible finding within the audited endpoint and exclusions. Lack of support for H1 is not invalid experimentation. No detected label, split, model-selection or primary prediction defect calls for a replacement experiment. This is not an unconditional dissertation-release approval or an institutional ethics determination.

Outstanding actions, in order:

1. Supply the issued committee approval and explicit practical-scope determination/amendment covering CT processing, training and intended outputs. The located form's blank committee fields do not establish that decision. Confirm historical timing; do not create retrospective approvals or dates.
2. Supply the original acquisition manifest/route, retrieval date, collection version where available, checksum provenance and usage/redistribution record. Current local inventory and independently verified hashes cannot recover missing historical facts.
3. Acknowledge the unresolved eight excluded anatomical groups and clustering coordinate convention. If expert adjudication is required for final approval, have it performed and keep any cohort-changing result separate from the exposed primary test evaluation.
4. Resolve the PDF/handover version label with the supervisor, using the recorded current PDF hash. The dissertation remains unchanged by this review.
5. Visually check the supplied separate hash-displaying interface correction and browser acceptance, or explicitly accept that requirement as incomplete. A generated HTML file and source inspection are not observed rendering or user testing. Keep the original five-pass route's incomplete clock and warm-up scope visible in resource captions; a new full pipeline benchmark would be supplemental after-test evidence, not a replacement of the original measurement.
6. Use the proposed Chapter 4 outline and traceability matrix to select tables and appendices; carry all applicable restrictions into captions and interpretation. Final author/supervisor and institutional acceptance remain outside automated verification.

At the user's instruction, no further search or requests for the missing approval/acquisition documents will be pursued during this task. They remain explicitly unresolved; this does not prevent reporting the verified numerical findings with their stated boundaries.

Any future correction affecting labels, splits, selection or predictions must preserve this run and disclose its effect. Model redesign after test inspection requires a new independent evaluation before a confirmatory claim; this test set must never be described as untouched again.
''')
    write('REPRODUCE.md','''# Reproducing the separate verification

Run from `C:\\Workspace\\research-project` using the retained Python 3.11 environment. Keep `artifacts/chapter3` and the reconnected raw drive unchanged. These commands write only the separate evidence directory (test temporary outputs are separate).

```powershell
.\\.venv\\Scripts\\python.exe scripts\\verify_results_evidence.py
.\\.venv\\Scripts\\python.exe scripts\\verify_clustering_evidence.py
.\\.venv\\Scripts\\python.exe scripts\\verify_source_evidence.py
.\\.venv\\Scripts\\python.exe scripts\\verify_representation_evidence.py
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts\\prepare_review_viewer.py
.\\.venv\\Scripts\\python.exe scripts\\assemble_results_review.py
.\\.venv\\Scripts\\python.exe scripts\\package_results_review.py
```

The source audit reads all retained source files for 689 prepared series and may take substantial USB-disk time; it does not run model fitting. The main verification uses 2,000 valid bootstrap draws. CUDA is required to reproduce the exact original randomized-network control; CPU initialization with the same seed creates a different control and is explicitly separate. Final-model prediction reload tolerances are rtol=1e-6/atol=1e-7; metric and interval agreement uses absolute 1e-12. Feature re-extraction uses rtol=atol=1e-8. The package script verifies the completed audit JSONs, immutable originals and local Markdown links, then CRC- and SHA-256-checks its archive.

The initial preservation command was `.\\.venv\\Scripts\\python.exe scripts\\preserve_results_evidence.py`. It intentionally refuses to overwrite its archive/manifest and should not be rerun in place. Its archive was read back, CRC-tested and every archived file checked against independently calculated SHA-256. End-of-review checks verify both artifacts and original source files remain unchanged.

`audit_execution.log`, `clustering_execution.log`, `source_execution.log`, `representation_execution.log` and `test_execution.log` retain completed command output. `audit_summary.json` must show `execution_complete=true`, not merely zero failed checks. Additional audit JSONs must independently be complete before their results are cited. No notebook/kernel rerun, experiment training, Git commit or push is part of these commands.

PDF reading used the system pdfplumber and workspace-isolated pypdf/pypdfium2 in `.build/evidence-pdf-deps`. These are review dependencies, not changes to the frozen experiment environment. The scanned rubric/template pages were rendered and visually read. Source documents and the dissertation were not changed.

File map: `tables/` contains machine-readable recomputed and audit outputs; `figures/` contains original-copy explanation/preprocessing panels and newly rendered saved-prediction plots; `prototype/` contains a freshly generated validation-case HTML viewer; `preservation/` contains the full original artifact archive and a snapshot of every tracked repository file, both with hashes. A copy of the reproduction scripts accompanies the review package under `reproduce_scripts/`; to run those copies, place them under the repository's `scripts/` directory so that relative project-root resolution remains correct. The self-contained review ZIP includes the preservation ZIPs and manifests. It does not include raw CT storage, the Python environment installation or proof of historical acquisition/approval.
''')
    write('README.md','''# Results evidence package - 12 November 2026

Start with the [evidence inventory](EVIDENCE_INVENTORY.md), then read the [separate implementation review](IMPLEMENTATION_REVIEW.md) and [readiness decision](READINESS.md).

- [Proposed Chapter 4 structure](CHAPTER4_STRUCTURE.md)
- [Objective/question traceability](TRACEABILITY.md)
- [Discrepancies and protocol deviations](DISCREPANCY_LOG.md)
- [Reproducible commands and evidence map](REPRODUCE.md)
- [Machine-readable verification checks](tables/verification_checks.csv)
- [Original preservation manifest](preservation/original_hashes.json)
- [Primary metric table](tables/metrics.csv) and [paired comparison](tables/primary_comparison.csv)

The current repository remains at the earlier review commit. The original experiment and revised dissertation are unchanged. This package adds verification after test inspection without refitting the experiment, replacing predictions, or drafting Chapters 5-6. The target is low/high radiologist-rated nodule malignancy suspicion; patient identity supplies independence for splitting and uncertainty.

Predictive calculations are reproducible. Final dissertation release remains conditional on unresolved evidence and acceptance requirements listed in READINESS.md. This local package includes patient-linked research artifacts; it has not been published or uploaded.
''')
    scripts=OUT/'reproduce_scripts';scripts.mkdir(exist_ok=True)
    for p in (ROOT/'scripts').glob('*results*evidence.py'):shutil.copyfile(p,scripts/p.name)
    for name in ['verify_clustering_evidence.py','verify_source_evidence.py','verify_representation_evidence.py','prepare_review_viewer.py','assemble_results_review.py','package_results_review.py']:
        shutil.copyfile(ROOT/'scripts'/name,scripts/name)
    figures=[]
    for name in ['roc.png','precision_recall.png','calibration.png','confusion_cnn.png','confusion_logistic_regression.png','confusion_random_forest.png','confusion_svm.png']:
        figures.append((f'figures/predictive/{name}','Re-rendered after test inspection from unchanged saved test predictions; 90 nodules from 56 patients; high suspicion positive; frozen operating thresholds for confusion matrices.'))
    for p in sorted((OUT/'figures/preprocessing').glob('*.png')):
        figures.append((p.relative_to(OUT).as_posix(),'Original preselected development QA panel: nearest native plane, standardized HU/contour and the two matched CNN channels. Nominal plane difference and scale are shown; not expert adjudication.'))
    for p in sorted((OUT/'figures/development_explanations').glob('*.png')):
        figures.append((p.relative_to(OUT).as_posix(),'Original development example, six sorted cases per class; trained/randomized high-suspicion-logit Grad-CAM. Neither map validates diagnostic reasoning.'))
    errors=frame('structured_errors.csv')
    for p in sorted((OUT/'figures/test_errors').glob('*.png')):
        reason=errors.loc[errors.nodule_id==p.stem,'selection_reason'].iloc[0]
        figures.append((p.relative_to(OUT).as_posix(),'Original structured post-evaluation test description; union of first five sorted nodule IDs per CNN/RF confusion category. '+reason))
    figures.append(('figures/learning_curves.png','Original development-only selected-configuration learning curves; no primary test model replacement.'))
    write('FIGURE_INDEX.md','# Figure and panel index\n\nEvery panel uses radiologist-rated suspicion as the reference. Original panels were copied byte-for-byte; newly rendered predictive plots use the unchanged saved predictions.\n\n'+'\n\n'.join('- ['+p+']('+p+') - '+caption for p,caption in figures)+'\n\nVisual QA during this review: all nine source/preprocessing panels, all seven predictive figures and a representative trained/randomized explanation panel were inspected. Other explanation/error panels are supplied and indexed, with their numerical and selection checks; this is not an expert or clinician review. Browser rendering of the prototype was not observed.')
    write('VISUAL_AND_INTERFACE_QA.md','''# Visual and interface verification boundary

The three rubric pages and nine template pages were rendered and read visually because their text extraction is empty. The revised PDF's method/objective sections were read from extracted text; page count is 80. The located ethics form was rendered to confirm its committee fields are blank, then left unresolved at the user's request.

All nine retained source/preprocessing panels and all seven saved-prediction figures were visually inspected: axes/labels and confusion counts are readable, the selected masks fit the displayed crop, masked intensity contains no exterior context, and native versus standardized differences are consistent with differing plane centres and interpolation. This is visual implementation QA, not clinical interpretation or proof of every source contour's correctness.

The structured error union and all twelve randomisation checks were verified numerically. A representative explanation panel was visually inspected; the other original panels remain indexed and supplied. No clinician explanation assessment was performed.

The frozen viewer passed programmatic prepared-case generation, invalid-input checks and research-label checks. A separate provenance-displaying copy retains its prediction table byte-for-byte. The browser-access host was unavailable; a fallback native-browser launch did not complete because its app approval timed out. Therefore no browser screenshot, rendered-layout acceptance or usability acceptance is claimed. This is a concrete remaining acceptance action, not a claim that the generated HTML failed.
''')
    print('Review documents assembled',flush=True)


if __name__=='__main__':main()
