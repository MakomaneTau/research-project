# Chapter 4 experimental evidence

Endpoint: low versus high radiologist-rated malignancy suspicion in pre-annotated LIDC-IDRI nodules.

Common cohort: 545 nodules from 368 patients.

## Frozen-test model comparison

| n_nodules | positive_nodules | prevalence | threshold | tp | fn | tn | fp | roc_auc | average_precision | accuracy | balanced_accuracy | sensitivity | specificity | precision | f1 | brier_score | model | n_patients | weighting | threshold_basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90 | 32 | 0.3556 | 0.5146 | 24.0000 | 8.0000 | 57.0000 | 1.0000 | 0.9402 | 0.9322 | 0.9000 | 0.8664 | 0.7500 | 0.9828 | 0.9600 | 0.8421 | 0.2524 | cnn | 56 | nodule | native_score |
| 90 | 32 | 0.3556 | 0.6916 | 23.0000 | 9.0000 | 57.0000 | 1.0000 | 0.9655 | 0.9606 | 0.8889 | 0.8508 | 0.7188 | 0.9828 | 0.9583 | 0.8214 | 0.0580 | logistic_regression | 56 | nodule | native_score |
| 90 | 32 | 0.3556 | 0.6420 | 27.0000 | 5.0000 | 57.0000 | 1.0000 | 0.9658 | 0.9631 | 0.9333 | 0.9133 | 0.8438 | 0.9828 | 0.9643 | 0.9000 | 0.0552 | random_forest | 56 | nodule | native_score |
| 90 | 32 | 0.3556 | 0.2562 | 28.0000 | 4.0000 | 56.0000 | 2.0000 | 0.9580 | 0.9530 | 0.9333 | 0.9203 | 0.8750 | 0.9655 | 0.9333 | 0.9032 | 0.0528 | svm | 56 | nodule | native_score |

## Primary paired comparison

```json
{
  "comparison": "cnn minus random_forest",
  "primary_metric": "roc_auc",
  "positive_class": "high radiologist-rated suspicion",
  "unit": "nodule",
  "resampling_unit": "patient",
  "interval": "95% percentile patient-cluster bootstrap",
  "seed": 42,
  "requested_replicates": 2000,
  "valid_replicates": 2000,
  "attempts": 2000,
  "rejected_single_class": 0,
  "reliable": true,
  "interpretation": "favors_selected_traditional_model",
  "differences": {
    "roc_auc": {
      "estimate": -0.02559267241379304,
      "lower_95": -0.05909453626867785,
      "upper_95": -0.0024786246348216176,
      "valid_metric_replicates": 2000
    },
    "sensitivity": {
      "estimate": -0.09375,
      "lower_95": -0.2413793103448275,
      "upper_95": 0.03228494623655907,
      "valid_metric_replicates": 2000
    },
    "specificity": {
      "estimate": 0.0,
      "lower_95": -0.050000000000000044,
      "upper_95": 0.054545454545454564,
      "valid_metric_replicates": 2000
    }
  },
  "uncertainty_scope": "Test-patient sampling conditional on fixed models and thresholds; excludes training/model-selection variability"
}
```

## Evidence files

- `cohort_summary.json`: cohort and partition denominators.
- `training/cv_results.csv`: all candidate/fold scores and failures.
- `results/test_predictions.csv`: traceable per-nodule predictions.
- `results/metric_intervals.csv`: patient-bootstrap uncertainty.
- `results/metrics_patient_weighted.csv` and `subgroup_metrics.csv`: secondary test summaries.
- `secondary/development_metrics.csv`: learning, weighting, augmentation, label and seed analyses.
- `secondary/mask_sensitivity.csv`: alternative-mask results and invalid alternatives.
- `explanations/`: coefficients, permutation importance and parameter-randomization checks.
- `test_explanations/test_error_panels/`: deterministic shared test-case review.
- `resources/`: measured computational costs.

The comparison estimates performance on the frozen internal cohort. Reader ratings are proxy labels; the experiment does not establish pathology-confirmed diagnosis, full-scan detection or clinical transfer.
