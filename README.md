# LIDC-IDRI model comparison

Implementation of the dissertation's Chapter 3 experiment: Logistic Regression,
RBF SVM, Random Forest and a compact two-channel CNN classify **low versus high
radiologist-rated nodule malignancy suspicion**. Expert annotations are supplied
inputs. This is an internal research comparison, not full-scan detection or
pathology-confirmed cancer diagnosis.

- [Open the demo](#open-the-demo)
- [Reproduce the experiment](#reproduce-the-experiment)

## Open the demo

For the Eduvos submission, download the `dissertation-v1-examiner-demo.zip`
attachment, extract it, and open `prepared_case.html` in your browser. The file
contains the axial CT crop, supplied consensus contour and four model decisions
for one validation nodule. It runs offline without Python, a dataset download
or model training.

## Recorded experiment

The completed predictive evaluation uses 545 nodules from 368 patients, including
90 final-test nodules from 56 patients. Test ROC-AUC is 0.9658 for Random Forest,
0.9655 for Logistic Regression, 0.9580 for SVM and 0.9402 for the compact CNN.
Random Forest was selected as the traditional comparator using training-only
cross-validation. The primary CNN-minus-RF AUC difference is -0.0256, with a
95% patient-bootstrap interval of [-0.0591, -0.0025], favoring RF in this cohort.

The CNN's native probabilities are poorly calibrated (Brier score 0.2524).
All four models fall below the validation sensitivity target on the final test.
These limitations belong alongside the discrimination results.

The [Chapter 4 results report](docs/CHAPTER4_RESULTS.md) gives the aggregate
results and uncertainty intervals. The examiner ZIP includes the original
`training/cv_results.csv` and `training/selection.json`, which record the
training-only selection of Random Forest. The executed notebook is available
at the recorded implementation commit.

## Reproduce the experiment

Start from the [recorded implementation commit](https://github.com/MakomaneTau/research-project/tree/92c854a9a69828624bc13a2999b4b8529e31d8de)
(`dissertation-v1`). It includes the source, configuration, environment lock,
tests and executed notebook used for the dissertation. Set the LIDC-IDRI data
location for your machine and follow the environment and workflow below. Use
the pinned checkout for the tracked-file hashes in the original completion
manifest.

### Environment

Use an isolated 64-bit Python 3.11 environment. The workstation has an existing
interpreter at `C:\Users\ponts\AppData\Local\Programs\Python\Python311\python.exe`.
The original Python installations and raw data are not modified.

`requirements-windows.lock.txt` records the verified package versions. For a
reproduction, install that lock, install this package with `--no-deps -e .`, then
build the pinned PyRadiomics wheel as described below. The commands that follow
also show how the isolated environment is bootstrapped.

```powershell
& 'C:\Users\ponts\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'setuptools>=68,<81' wheel numpy==1.26.4
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip install PyWavelets pykwalify six
```

PyRadiomics 3.1.0 has no published CPython 3.11 Windows wheel. Install Microsoft
Visual Studio 2022 C++ Build Tools and an x64 Windows SDK, then run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_pyradiomics.ps1
.\.venv\Scripts\python.exe scripts/environment_smoke.py
.\.venv\Scripts\python.exe -m pytest
```

The build script verifies official release commit
`6a761c4eb01905cfcb1139b0b158547c55edd4df` and records the resulting wheel hash.
It records packaging-only corrections for the release's static version and
v-prefixed tag configuration, verifies that scientific source files are unchanged,
and retains the patch and wheel hashes. The environment
check requires compiled feature extraction, annotation access and a CUDA
forward/backward pass at batch size 32. A narrow compatibility adapter restores
removed NumPy builtin aliases only inside `pylidc` module globals.

## Run and resume

Configure raw data and artifact paths in [configs/chapter3.yaml](configs/chapter3.yaml).
`pylidc` must already resolve the acquired CT files beneath the configured data root.
No data download is performed. All generated data, models and reports stay under
`artifacts/chapter3`, outside Git.

Check recorded progress with `python -m lidc_experiment status`. The status command
does not run preprocessing, train models or access final-test predictions.

```powershell
.\.venv\Scripts\python.exe -m lidc_experiment prepare
.\.venv\Scripts\python.exe -m lidc_experiment features
.\.venv\Scripts\python.exe -m lidc_experiment split
```

Inspect `artifacts/chapter3/data/development_overlays.png`, then record the actual
review, including the native-source comparisons in `data/source_panels/` and
recorded exclusions. Any unresolved extraction problem must be fixed and the affected inputs
regenerated before model development.

```powershell
.\.venv\Scripts\python.exe -m lidc_experiment review --review-note 'Describe the inspected contour alignment, field of view and any findings'
.\.venv\Scripts\python.exe -m lidc_experiment tune
.\.venv\Scripts\python.exe -m lidc_experiment fit
.\.venv\Scripts\python.exe -m lidc_experiment thresholds
.\.venv\Scripts\python.exe -m lidc_experiment secondary
.\.venv\Scripts\python.exe -m lidc_experiment explain
.\.venv\Scripts\python.exe -m lidc_experiment check
.\.venv\Scripts\python.exe -m lidc_experiment freeze
.\.venv\Scripts\python.exe -m lidc_experiment evaluate
.\.venv\Scripts\python.exe -m lidc_experiment benchmark
.\.venv\Scripts\python.exe -m lidc_experiment report
```

`run` executes the same stages sequentially and stops after the first split for
development-overlay inspection, before expensive fitting. Record the review,
then resume `run`. Stages cache compatible
derived cases and fits; changed sources, configurations or feature values refuse
incompatible reuse. Use a fresh output directory for a changed protocol. Reduced
grids, epochs and patient limits require an explicit `profile: smoke` configuration
and a separate output directory.

## Protocol and artifacts

- One eligible series per patient; spacing <=2.5 mm; >=2 readers; mean annotation
  diameter >=3 mm. Mean ratings <=2 become label 0; >=4 become label 1; intermediate
  nodules remain available for development-only median-label analysis.
- Linear HU and nearest-neighbor reader-mask resampling share 1 mm physical
  geometry. A >=50% consensus determines the maximal-area axial slice, with
  centroid-distance and lower-index tie breaking. Both pathways use an intact
  64x64 mm contour/crop. HU is clipped to [-1000,400]; the CNN receives masked
  normalized intensity and the binary mask. Original 2D radiomics uses 25-HU bins.
- The common binary cohort is frozen after geometry and extraction QC. Patients
  receive approximately 70/15/15 training/threshold-validation/test allocation;
  three stored training folds are shared by all candidate models.
- Every learned feature transform and SVM calibration fold uses training patients
  only. The full declared grids contain 12 candidates per model. CNN inner early
  stopping is followed by an outer-training refit; final fitting uses the selected
  median epoch count. Threshold validation is separate from model selection.
- `secondary/` holds learning curves, weighting/augmentation alternatives, median
  labels, seeds and alternative-mask checks. No secondary result replaces the
  primary model after test inspection.
- `frozen.json` seals configuration, source code, cohort/features, final models,
  thresholds and completion checks. Evaluation verifies these hashes and writes
  paired predictions and 2,000-replicate patient-bootstrap uncertainty.
- `resources/` separates search/final fitting, process/GPU memory, model size,
  batch-one CPU inference and decoded-CT pipeline latency. Pipeline transformations
  use in-memory arrays; CT disk decoding is timed separately. Ten model-only
  warm-ups precede five complete test-list passes. Decoded-CT route costs sum
  separately timed shared work and branch processing. A supplemental one-pass
  component recorder saves preparation, extraction, representation assembly and
  prediction costs, with ten complete-path warm-ups and frozen-input parity
  checks. Five complete pipeline passes can take substantially longer than
  model-only inference. Measurement-only amendments and original manifests are
  retained for the recorded run.

Open [notebooks/chapter3_experiment.ipynb](notebooks/chapter3_experiment.ipynb) to
inspect recorded tables and figures. Missing experiments are displayed as missing,
not filled with example scores. `report` creates `CHAPTER4_RESULTS.md` from saved
predictions. `view --nodule-id ID` creates a local HTML view for a prepared annotated
case; test-case prediction access requires completed final evaluation.

## Source-to-method links

The feature definitions and provenance follow
[van Griethuysen et al.](https://doi.org/10.1158/0008-5472.CAN-17-0339),
[IBSI](https://doi.org/10.1148/radiol.2020191145) and the
[PyRadiomics configuration documentation](https://pyradiomics.readthedocs.io/en/latest/customization.html).
Grouped validation follows the
[scikit-learn group-split interface](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
Explanations use [Grad-CAM](https://arxiv.org/abs/1610.02391) and the
[parameter-randomization sanity-check rationale](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).
The precise crop, grids, CNN architecture, split and threshold target are declared
study choices from Chapter 3, not claims of reproducing published hyperparameters.

## Verification boundary

Synthetic tests verify label boundaries, spatial transforms, patient/source
independence, train-only feature fitting, calibration, metric arithmetic,
bootstrap multiplicity, reload consistency and four-model integration.
These tests do not establish predictive performance. Real cohort counts and model
scores are valid only after their corresponding recorded stages finish.

## Research outputs

- *Comparing Radiomics and Deep Learning for Pulmonary Nodule Suspicion
  Classification* (A patient-grouped evaluation using LIDC-IDRI CT images).
  `ITRPA4-34 - Mini Dissertation - Midrand - con-1471898-j4q5.pdf`
