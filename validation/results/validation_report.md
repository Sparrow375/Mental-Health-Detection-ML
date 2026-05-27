# MHealth System Validation Report
### StudentLife Dataset — 2026-05-25 23:58

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
| **AUC-ROC** | 0.8361 |
| **AUC-PR** | 0.8144 |
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
![ROC Curve](F:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\roc_curve.png)

### Score Distributions
![Score Distributions](F:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\score_distributions.png)

### PHQ-9 Correlation
![PHQ-9 Correlation](F:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\correlation_scatter.png)

### Confusion Matrix
![Confusion Matrix](F:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\confusion_matrix.png)

### Student Timeseries
![Student Timeseries](F:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\student_timeseries.png)

### Parameter Sensitivity
![Parameter Sensitivity](F:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML\validation\results\plots\parameter_sensitivity.png)

## Anomaly Score ↔ Depression Correlation

| Metric | Value | p-value |
|--------|-------|---------|
| Pearson Anomaly | 0.3981 | 0.006139 |
| Spearman Anomaly | 0.41 | 0.004657 |
| Pearson Evidence | 0.3709 | 0.01117 |
| Spearman Evidence | 0.3049 | 0.039362 |
| Pointbiserial Anomaly | 0.4101 | 0.004643 |
| Pointbiserial Evidence | 0.4864 | 0.000609 |

- **Cohens D Anomaly**: 1.0887
- **Cohens D Evidence**: 1.3478
- **Mann-Whitney U**: U=301.0, p=0.000666

## Optimal Operating Points

| Metric | Threshold | Sensitivity | Specificity | F1 | Youden's J |
|--------|-----------|-------------|-------------|-----|-----------|
| mean_anomaly_score | 0.5128 | 0.7 | 1.0 | 0.8235 | 0.7 |
| peak_evidence | 94.6603 | 0.8 | 0.8056 | 0.64 | 0.6056 |
| peak_sustained_days | 51.8384 | 0.9 | 0.7222 | 0.6207 | 0.6222 |
| anomaly_day_ratio | 0.947 | 0.8 | 0.7222 | 0.5714 | 0.5222 |

## Best Configuration

**Config Name**: `R071_exp_0.8_0.2`
**Ground Truth Strategy**: `max`
**Composite Score**: `0.8419`

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
| u00 ✅ | 2 | 3 | ✗ | ✗ | 0.2547 | 17.9262 | 120 |
| u01 ✅ | 5 | 4 | ✗ | ✗ | 0.426 | 25.5653 | 66 |
| u02 ❌ | 13 | 5 | ✗ | ✓ | 0.6065 | 184.1501 | 70 |
| u03 ✅ | 2 | 4 | ✗ | ✗ | 0.4666 | 64.2749 | 57 |
| u04 ✅ | 6 | 8 | ✗ | ✗ | 0.4131 | 15.4665 | 62 |
| u05 ✅ | 2 | 0 | ✗ | ✗ | 0.4575 | 72.7998 | 68 |
| u07 ✅ | 7 | 8 | ✗ | ✗ | 0.4897 | 80.2812 | 53 |
| u08 ✅ | 5 | None | ✗ | ✗ | 0.4118 | 32.6973 | 69 |
| u09 ✅ | 4 | 2 | ✗ | ✗ | 0.4475 | 94.4378 | 73 |
| u10 ✅ | 0 | 4 | ✗ | ✗ | 0.428 | 33.0663 | 72 |
| u12 ✅ | 1 | None | ✗ | ✗ | 0.4197 | 42.7415 | 73 |
| u13 ✅ | 4 | None | ✗ | ✗ | 0.4507 | 145.4451 | 100 |
| u14 ✅ | 1 | 3 | ✗ | ✗ | 0.4781 | 104.1474 | 68 |
| u15 ✅ | 3 | 1 | ✗ | ✗ | 0.4578 | 58.2821 | 50 |
| u16 ❌ | 6 | 12 | ✓ | ✗ | 0.4205 | 45.9796 | 68 |
| u17 ✅ | 13 | 18 | ✓ | ✓ | 0.564 | 132.0321 | 71 |
| u18 ✅ | 15 | 12 | ✓ | ✓ | 0.5364 | 86.0402 | 63 |
| u19 ✅ | 5 | 4 | ✗ | ✗ | 0.4431 | 46.9522 | 71 |
| u20 ✅ | 8 | 8 | ✗ | ✗ | 0.4939 | 63.7981 | 54 |
| u22 ✅ | 3 | None | ✗ | ✗ | 0.4838 | 134.9592 | 69 |
| u23 ✅ | 11 | 21 | ✓ | ✓ | 0.5309 | 106.4684 | 62 |
| u24 ✅ | 5 | 7 | ✗ | ✗ | 0.4684 | 32.2039 | 36 |
| u25 — | None | None | ? | ✗ | 0.4957 | 82.3348 | 52 |
| u27 ✅ | 5 | 7 | ✗ | ✗ | 0.4518 | 86.6488 | 72 |
| u30 ✅ | 1 | 0 | ✗ | ✗ | 0.4589 | 85.4165 | 68 |
| u31 ❌ | 12 | 5 | ✗ | ✓ | 0.6021 | 166.3115 | 69 |
| u32 ✅ | 4 | 2 | ✗ | ✗ | 0.5085 | 133.9601 | 70 |
| u33 ✅ | 23 | 25 | ✓ | ✓ | 0.5572 | 104.9175 | 59 |
| u34 ✅ | 3 | 6 | ✗ | ✗ | 0.5004 | 83.0773 | 53 |
| u35 ✅ | 7 | 7 | ✗ | ✗ | 0.4852 | 139.31 | 69 |
| u36 ✅ | 2 | 1 | ✗ | ✗ | 0.4711 | 147.2827 | 79 |
| u39 ✅ | 3 | None | ✗ | ✗ | 0.4374 | 11.5242 | 28 |
| u41 — | None | None | ? | ✗ | 0.4843 | 86.5918 | 57 |
| u42 ✅ | 1 | 0 | ✗ | ✗ | 0.508 | 61.8196 | 54 |
| u43 ✅ | 7 | 4 | ✗ | ✗ | 0.4898 | 115.4829 | 62 |
| u44 ✅ | 1 | 2 | ✗ | ✗ | 0.468 | 90.5437 | 65 |
| u45 ✅ | 7 | 2 | ✗ | ✗ | 0.4922 | 89.7149 | 54 |
| u46 ❌ | 10 | None | ✗ | ✓ | 0.5707 | 122.4895 | 61 |
| u47 ✅ | 5 | 1 | ✗ | ✗ | 0.1649 | 11.7766 | 138 |
| u49 ✅ | 2 | 8 | ✗ | ✗ | 0.2225 | 76.9526 | 159 |
| u50 ✅ | 7 | None | ✗ | ✗ | 0.4743 | 55.6123 | 50 |
| u51 ✅ | 1 | 0 | ✗ | ✗ | 0.2397 | 3.3443 | 101 |
| u52 ❌ | 12 | 15 | ✓ | ✗ | 0.3159 | 124.6982 | 181 |
| u53 ❌ | 8 | 11 | ✓ | ✗ | 0.4937 | 131.8962 | 68 |
| u54 — | None | None | ? | ✗ | 0.2276 | 6.0413 | 77 |
| u56 ✅ | 2 | 3 | ✗ | ✗ | 0.226 | 29.3151 | 124 |
| u57 ✅ | 0 | None | ✗ | ✗ | 0.2167 | 42.7359 | 159 |
| u58 ✅ | 5 | 8 | ✗ | ✗ | 0.3084 | 17.581 | 85 |
| u59 ✅ | 5 | 7 | ✗ | ✗ | 0.4257 | 27.5285 | 77 |

## Trial Comparison Summary

### Strategy: `post`
- Best: `T007_strat_mean_anomaly_bd5_psc0.52_overridesTrue`
- Sensitivity: 0.7143
- Specificity: 0.8387
- AUC-ROC: 0.6682
- F1: 0.5882

### Strategy: `any`
- Best: `T007_strat_mean_anomaly_bd5_psc0.52_overridesTrue`
- Sensitivity: 0.75
- Specificity: 0.8684
- AUC-ROC: 0.7237
- F1: 0.6316

### Strategy: `max`
- Best: `T007_strat_mean_anomaly_bd5_psc0.52_overridesTrue`
- Sensitivity: 0.8
- Specificity: 0.9167
- AUC-ROC: 0.8111
- F1: 0.7619

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
