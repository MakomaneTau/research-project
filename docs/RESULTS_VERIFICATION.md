# Results verification and local evidence package

The 12 November 2026 review independently checked the retained experiment at
commit `92c854a9a69828624bc13a2999b4b8529e31d8de`. It preserved the original
scientific artifacts before recalculation. No experiment model was retrained,
primary prediction replaced, or dissertation chapter edited. Verification after
test inspection is additional evidence; it does not make the test set untouched
again.

## Verified findings

- The common binary cohort contains 545 nodules from 368 patients. Training,
  threshold validation and test contain 379/76/90 nodules from 257/55/56 patients.
- Patient, source-series and source-image overlap between partitions is zero;
  training-fold patients are also disjoint.
- All 144 candidate-fold records and the four final models trace to their
  prescribed training observations. The CNN's selected fold epochs, 1/9/1,
  produce the protocol's one-epoch final fit.
- Saved test ROC-AUCs reproduce: Random Forest 0.9658, Logistic Regression
  0.9655, SVM 0.9580 and CNN 0.9402. The CNN-minus-CV-selected-RF difference is
  -0.0255926724, with a 95% patient-cluster bootstrap interval of
  [-0.0590945363, -0.0024786246]. Seed 42 gives 2,000 valid replicates, with no
  single-class rejections and repeated patients' multiplicities preserved.
- All 193 saved-evidence audit checks passed. The separate source audit matched
  182,191 retained source-image hashes across 689 prepared series, reading
  95,896,473,236 bytes. Nine development QA reconstructions were identical and
  the four recorded geometry exclusions reproduced.
- Re-extraction matched all 1,594 feature rows, all 464 alternate-mask prediction
  rows and validation permutation-importance results.

These are nodule predictions of low/high **radiologist-rated malignancy
suspicion**. They are not pathology-confirmed cancer outcomes or patient-level
clinical diagnoses. All models fall below 0.90 sensitivity on the test set. The
CNN's Brier score is approximately 0.2524, despite good ranking discrimination.
An unsupported CNN-superiority hypothesis is a legitimate result.

## Evidence tools

The scripts are dated entry points for this preserved experiment, using
`artifacts/chapter3` and writing to `results_evidence/2026-11-12`. Original input
paths and the recorded Python 3.11 environment are required. The main audit also
expects the reviewed PDF/handover files at their recorded workstation paths.
CUDA is needed to reproduce the original randomized-network control exactly;
CPU and CUDA seeded parameter initialization produce different controls.

| Script | Purpose |
| --- | --- |
| `scripts/preserve_results_evidence.py` | Archive and hash original artifacts; refuse missing, empty or already-preserved inputs |
| `scripts/verify_results_evidence.py` | Independently check cohort, folds, transformations, model reloads, thresholds, metrics, bootstrap, robustness and prototype contracts; return failure if any check fails |
| `scripts/verify_clustering_evidence.py` | Reconstruct annotation selection and compare grouping state machines, with sampled independent distance checks |
| `scripts/verify_source_evidence.py` | Inventory original storage, verify source hashes, reconstruct nine QA cases and recheck geometry exclusions |
| `scripts/verify_representation_evidence.py` | Re-extract features, alternate-mask predictions and permutation importance without refitting |
| `scripts/prepare_review_viewer.py` | Generate a separately labelled viewer copy with exact model/configuration hashes and unchanged predictions |
| `scripts/assemble_results_review.py` | Assemble the inventory, review, Chapter 4 outline, traceability, deviations, readiness and figure index |
| `scripts/package_results_review.py` | Verify immutable originals, completed audits and local links, then seal the package with ZIP and SHA-256 checks |

Run the preservation script **once**, before the remaining commands. It
intentionally refuses to overwrite an existing archive. The other commands can
refresh the separate review outputs; preserve a completed delivery before doing
so if its exact historical bytes must remain available.

```powershell
.\.venv\Scripts\python.exe scripts\preserve_results_evidence.py
.\.venv\Scripts\python.exe scripts\verify_results_evidence.py
.\.venv\Scripts\python.exe scripts\verify_clustering_evidence.py
.\.venv\Scripts\python.exe scripts\verify_source_evidence.py
.\.venv\Scripts\python.exe scripts\verify_representation_evidence.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\prepare_review_viewer.py
.\.venv\Scripts\python.exe scripts\assemble_results_review.py
.\.venv\Scripts\python.exe scripts\package_results_review.py
```

The source audit is disk-intensive and took approximately forty minutes on the
retained USB storage. Recalculation uses preserved predictions, not model
retraining. Tests use temporary synthetic data where fitting is necessary for
software checks.

## Readiness and measurement boundaries

The primary numerical comparison is **ready with disclosed limitations**.
Model-selection traceability and representation checks are ready. Unconditional
dissertation release remains not ready because historical acquisition/usage and
issued practical-scope approval evidence are unresolved, and browser acceptance
was not observed. The user instructed the review to leave those missing
documents unresolved; no approval or refusal is inferred.

Additional restrictions are recorded in the local deviation register:

- Eight ambiguous selected-series annotation groups remain quarantined without
  expert anatomical adjudication. Equivalence to installed `pylidc` grouping is
  distinct from anatomical validity. The algorithm retains its upstream contour
  index-coordinate distance convention.
- The original five-pass resource route is a partial component sum. Its clocks
  omit tensor/dataframe assembly and it lacks dedicated full-path warm-ups.
  Ten warm-ups apply to model-only inference; the supplemental component run has
  ten full-path warm-ups but only one timed pass. Keep these measurements separate.
- Training uses different CPU/GPU devices; LR/SVM training BLAS threadpool state
  was not captured. Memory is sampled absolute process RSS or PyTorch allocated
  memory, not isolated whole-service memory.
- Median labels define a changed development task/cohort. Mask score changes
  must be compared within each model's native score scale.
- Randomized Grad-CAM controls remain substantially similar to trained maps;
  they do not establish learned disease localization, clinician trust or clinical
  interpretability superiority.
- The original viewer is unchanged. A separately labelled provenance-displaying
  copy exists, but browser rendering and clinician/usability testing are unverified.

The local package contains the complete review, a proposed Chapter 4 structure,
objective/question traceability, exact outstanding actions, machine-readable
checks, figures, original artifact preservation and a repository snapshot. It
does not draft Chapters 5-6 or establish deployment readiness.

## Repository and sharing boundary

Both `artifacts/` and `results_evidence/` remain ignored by Git. They include
fitted models, patient-linked predictions/manifests, supplied contours, image
panels, source documents and large archives. This repository commits the audit
tools and this aggregate summary. Committing code does not publish the local
evidence package or establish permission to redistribute its contents.

The original completed delivery remains available locally at
`results_evidence/2026-11-12/review_package.zip`, with a SHA-256 companion file and
per-file checksums. Historical package snapshots identify the repository state
at their creation; later commits of the verification tools do not change that
experiment provenance.
