# MHealth System Validation Report
### StudentLife Dataset — 2026-06-27 22:53

---

## Executive Summary

The MHealth anomaly detection system was validated against the StudentLife dataset (49 students, 46 with PHQ-9 ground truth).

| Metric | Value |
|--------|-------|
| **Sensitivity (Recall)** | **0.7** |
| **Specificity** | **1.0** |
| **PPV (Precision)** | 1.0 |
| **NPV** | 0.9231 |
| **F1 Score** | 0.8235 |
| **Balanced Accuracy** | 0.85 |
| **AUC-ROC** | 0.8444 |
| **AUC-PR** | 0.8222 |
| **Cohen's Kappa** | 0.785 |
| **MCC** | 0.8038 |
| **Youden's J** | 0.7 |

## Confusion Matrix

| | Predicted Non-Depressed | Predicted Depressed |
|---|---|---|
| **Actual Non-Depressed** | 36 (TN) | 0 (FP) |
| **Actual Depressed** | 3 (FN) | 7 (TP) |

## Visual Analysis

### ROC Curve
![ROC Curve](f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\roc_curve.png)

### Score Distributions
![Score Distributions](f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\score_distributions.png)

### PHQ-9 Correlation
![PHQ-9 Correlation](f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\correlation_scatter.png)

### Confusion Matrix
![Confusion Matrix](f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\confusion_matrix.png)

### Student Timeseries
![Student Timeseries](f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\student_timeseries.png)

### Parameter Sensitivity
![Parameter Sensitivity](f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\parameter_sensitivity.png)

## Anomaly Score ↔ Depression Correlation

| Metric | Value | p-value |
|--------|-------|---------|
| Pearson Anomaly | 0.3804 | 0.009118 |
| Spearman Anomaly | 0.4254 | 0.003208 |
| Pearson Evidence | 0.3543 | 0.015686 |
| Spearman Evidence | 0.2866 | 0.053513 |
| Pointbiserial Anomaly | 0.3923 | 0.007001 |
| Pointbiserial Evidence | 0.4712 | 0.000949 |

- **Cohens D Anomaly**: 1.0317
- **Cohens D Evidence**: 1.293
- **Mann-Whitney U**: U=304.0, p=0.000503

## Optimal Operating Points

| Metric | Threshold | Sensitivity | Specificity | F1 | Youden's J |
|--------|-----------|-------------|-------------|-----|-----------|
| mean_anomaly_score | 0.5038 | 0.8 | 0.9167 | 0.7619 | 0.7167 |
| peak_evidence | 102.268 | 0.8 | 0.8056 | 0.64 | 0.6056 |
| peak_sustained_days | 51.8384 | 0.9 | 0.6944 | 0.6 | 0.5944 |
| anomaly_day_ratio | 0.9463 | 0.8 | 0.6389 | 0.5161 | 0.4389 |

## Best Configuration

**Config Name**: `R071_exp_0.8_0.2`
**Ground Truth Strategy**: `max`
**Composite Score**: `0.8436`

| Parameter | Value |
|-----------|-------|
| anomaly_threshold | 0.38 |
| baseline_days | 5 |
| clinical_overrides_enabled | False |
| compactness_N | 7 |
| compactness_threshold | 1.2 |
| evidence_compounding | 0.15 |
| evidence_decay | 0.88 |
| hybrid_mode | True |
| kappa_0 | 14.0 |
| l1_exponent | 0.8 |
| l2_exponent | 0.2 |
| prediction_evidence_threshold | 40.0 |
| prediction_score_threshold | 0.52 |
| prediction_strategy | mean_anomaly |
| prediction_sustained_threshold | 25 |

## Per-Student Results

| Student | PHQ-9 Pre | PHQ-9 Post | Depressed | Predicted | Mean Score | Peak Evidence | Days |
|---------|-----------|------------|-----------|-----------|------------|---------------|------|
| u00 ✅ | 2 | 3 | ✗ | ✗ | 0.2392 | 9.2118 | 120 |
| u01 ✅ | 5 | 4 | ✗ | ✗ | 0.4289 | 73.3125 | 66 |
| u02 ❌ | 13 | 5 | ✗ | ✓ | 0.6049 | 184.5956 | 70 |
| u03 ✅ | 2 | 4 | ✗ | ✗ | 0.4739 | 67.3565 | 57 |
| u04 ✅ | 6 | 8 | ✗ | ✗ | 0.418 | 26.8655 | 62 |
| u05 ✅ | 2 | 0 | ✗ | ✗ | 0.458 | 75.5415 | 68 |
| u07 ✅ | 7 | 8 | ✗ | ✗ | 0.4915 | 83.1411 | 53 |
| u08 ✅ | 5 | None | ✗ | ✗ | 0.4266 | 45.8635 | 69 |
| u09 ✅ | 4 | 2 | ✗ | ✗ | 0.4473 | 89.1994 | 73 |
| u10 ✅ | 0 | 4 | ✗ | ✗ | 0.4337 | 56.1599 | 72 |
| u12 ✅ | 1 | None | ✗ | ✗ | 0.4214 | 44.7997 | 73 |
| u13 ✅ | 4 | None | ✗ | ✗ | 0.448 | 143.1097 | 100 |
| u14 ✅ | 1 | 3 | ✗ | ✗ | 0.4747 | 119.7925 | 68 |
| u15 ✅ | 3 | 1 | ✗ | ✗ | 0.4627 | 60.1198 | 50 |
| u16 ❌ | 6 | 12 | ✓ | ✗ | 0.4299 | 51.6341 | 68 |
| u17 ✅ | 13 | 18 | ✓ | ✓ | 0.558 | 135.0554 | 71 |
| u18 ✅ | 15 | 12 | ✓ | ✓ | 0.5334 | 87.5244 | 63 |
| u19 ✅ | 5 | 4 | ✗ | ✗ | 0.4494 | 61.8154 | 71 |
| u20 ✅ | 8 | 8 | ✗ | ✗ | 0.5012 | 80.577 | 54 |
| u22 ✅ | 3 | None | ✗ | ✗ | 0.4876 | 145.4285 | 69 |
| u23 ✅ | 11 | 21 | ✓ | ✓ | 0.532 | 109.635 | 62 |
| u24 ✅ | 5 | 7 | ✗ | ✗ | 0.4809 | 36.487 | 36 |
| u25 — | None | None | ? | ✗ | 0.5007 | 82.163 | 52 |
| u27 ✅ | 5 | 7 | ✗ | ✗ | 0.4603 | 92.3695 | 72 |
| u30 ✅ | 1 | 0 | ✗ | ✗ | 0.4574 | 101.408 | 68 |
| u31 ❌ | 12 | 5 | ✗ | ✓ | 0.6019 | 173.4436 | 69 |
| u32 ✅ | 4 | 2 | ✗ | ✗ | 0.5104 | 136.6799 | 70 |
| u33 ✅ | 23 | 25 | ✓ | ✓ | 0.556 | 105.3174 | 59 |
| u34 ✅ | 3 | 6 | ✗ | ✗ | 0.505 | 89.4001 | 53 |
| u35 ✅ | 7 | 7 | ✗ | ✗ | 0.4946 | 138.4614 | 69 |
| u36 ✅ | 2 | 1 | ✗ | ✗ | 0.4713 | 145.7319 | 79 |
| u39 ✅ | 3 | None | ✗ | ✗ | 0.4618 | 15.6036 | 28 |
| u41 — | None | None | ? | ✗ | 0.4896 | 101.3336 | 57 |
| u42 ✅ | 1 | 0 | ✗ | ✗ | 0.5114 | 69.0007 | 54 |
| u43 ✅ | 7 | 4 | ✗ | ✗ | 0.4934 | 117.5513 | 62 |
| u44 ✅ | 1 | 2 | ✗ | ✗ | 0.4727 | 97.7227 | 65 |
| u45 ✅ | 7 | 2 | ✗ | ✗ | 0.5003 | 88.459 | 54 |
| u46 ❌ | 10 | None | ✗ | ✓ | 0.5735 | 124.0671 | 61 |
| u47 ✅ | 5 | 1 | ✗ | ✗ | 0.1499 | 9.0511 | 138 |
| u49 ✅ | 2 | 8 | ✗ | ✗ | 0.2104 | 60.2113 | 159 |
| u50 ✅ | 7 | None | ✗ | ✗ | 0.4797 | 66.3462 | 50 |
| u51 ✅ | 1 | 0 | ✗ | ✗ | 0.2394 | 3.4749 | 101 |
| u52 ❌ | 12 | 15 | ✓ | ✗ | 0.315 | 128.8634 | 181 |
| u53 ❌ | 8 | 11 | ✓ | ✗ | 0.5045 | 135.033 | 68 |
| u54 — | None | None | ? | ✗ | 0.2214 | 4.9403 | 77 |
| u56 ✅ | 2 | 3 | ✗ | ✗ | 0.2093 | 19.8959 | 124 |
| u57 ✅ | 0 | None | ✗ | ✗ | 0.2032 | 32.4554 | 159 |
| u58 ✅ | 5 | 8 | ✗ | ✗ | 0.2926 | 5.5733 | 85 |
| u59 ✅ | 5 | 7 | ✗ | ✗ | 0.4177 | 18.873 | 77 |

## Trial Comparison Summary

### Strategy: `post`
- Best: `T007_strat_mean_anomaly_bd5_psc0.52_overridesTrue`
- Sensitivity: 0.7143
- Specificity: 0.7419
- AUC-ROC: 0.6636
- F1: 0.5

### Strategy: `any`
- Best: `T007_strat_mean_anomaly_bd5_psc0.52_overridesTrue`
- Sensitivity: 0.75
- Specificity: 0.7895
- AUC-ROC: 0.7171
- F1: 0.5455

### Strategy: `max`
- Best: `T007_strat_mean_anomaly_bd5_psc0.52_overridesTrue`
- Sensitivity: 0.8
- Specificity: 0.8333
- AUC-ROC: 0.8056
- F1: 0.6667

## Methodology

### Data Processing
- StudentLife raw sensor CSVs transformed into daily PersonalityVector-compatible features
- GPS → displacement, entropy, homeTimeRatio (Grid-Cell Transition Method)
- phonelock + dark → sleep duration, wake/sleep times (3-Signal Sleep Fusion)
- activity inference → step count, active minutes
- call_log → calls/day, duration, unique contacts, conversation frequency
- conversation → social ratio (face-to-face proxy)
- audio inference → ambient activity proxy (daylight exposure)
- app_usage → app launches, notification proxy
- phonecharge → charge duration, charge regularity

### Simulation
- Data fed day-by-day through MHealth System 1 pipeline
- Bayesian baseline with rolling updates (hybrid mode)
- L1 Scorer: weighted z-scores + EWMA velocity
- Evidence Engine: compounding accumulation with decay
- Multi-signal prediction: evidence + sustained days + anomaly ratio + mean score

### Ground Truth
- PHQ-9 questionnaire (pre and post study)
- Clinical depression cutoff: PHQ-9 ≥ 10 (moderate depression)
- Multiple strategies tested: post-only, pre-only, max(pre,post), any available
