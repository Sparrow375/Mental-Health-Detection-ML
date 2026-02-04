# Early Risk Detection of Mental Health Disorders

## 1. Project Overview

Early detection of mental health disorders enables timely support and intervention before conditions become severe. Many individuals at risk do not voluntarily seek clinical help, and outward behavior may appear normal despite internal distress. This project proposes a **passive, risk-averse screening system** that analyzes **voice and behavioral indicators** to flag early warning signs. The system is designed as a **decision-support tool**, not a diagnostic system.

**Core principles**:

* Early **risk flagging**, not diagnosis
* **Pessimistic (risk-averse)** design to minimize false negatives
* **Privacy-aware**, modular, and future-compatible
* Emphasis on **interpretability and confidence**

---

## 2. Problem Statement

Traditional mental health assessments rely on self-reporting and clinical observation, which can delay identification of at-risk individuals. Automated analysis of everyday signals—especially voice and behavior—can reveal subtle deviations earlier. The challenge is to design a system that is accurate, ethical, explainable, and feasible within academic constraints.

---

## 3. Scope and Target Disorders

### Primary focus (Phase 1):

* Depression
* Anxiety disorders

### Future extension (out of scope for current implementation):

* Bipolar disorder
* Schizophrenia

Rationale: Depression and anxiety exhibit strong, well-documented voice and behavioral markers and have better data availability.

---

## 4. System Architecture (High-Level)

### Modular Design

The system is intentionally split into independent layers:

1. **Data Sensing Layer (Deferred)**

   * Smartphone-based passive data collection
   * Includes voice capture and behavioral metrics
   * Defined interfaces only; not implemented in this phase

2. **Intelligence Layer (Implemented)**

   * Feature engineering
   * Machine learning models
   * Risk scoring and confidence estimation

3. **Presentation Layer (Minimal Implementation)**

   * Simple dashboard
   * Risk bands, trends, and confidence indicators

This separation allows development of the core intelligence without dependency on mobile app development.

---

## 5. Data Inputs

### 5.1 Voice Features (Primary Signal)

Extracted using Librosa:

* MFCCs (vocal tract and articulation patterns)
* Fundamental frequency (pitch)
* Jitter (frequency instability)
* Shimmer (amplitude instability)
* Harmonics-to-Noise Ratio (HNR)

**Rationale**: Voice reflects emotional state, cognitive load, and neuromotor control.

---

### 5.2 Behavioral Parameters (Planned Inputs)

These are treated as **first-class inputs**, even if populated via simulation or datasets in the current phase.

Examples:

* Screen time
* Call frequency and duration
* Interaction frequency
* Sleep proxy (activity timing)
* Mobility variance (GPS-derived)
* Battery usage trends

**Current status**: Values provided via datasets, manual logs, or simulated proxies. Automatic phone sensing is deferred.

---

## 6. Input Contract and Missing Data Handling

### Input Schema

Each user-day record is represented as a structured vector containing:

* Voice feature set
* Behavioral parameter set

### Missing Data Strategy

* Conservative imputation for missing signals
* Increased uncertainty when behavioral data is absent
* Confidence score reduced proportional to data coverage

This ensures robustness and honest uncertainty reporting.

---

## 7. Feature Engineering

### Voice-Based Features

* Statistical aggregates over time windows (mean, variance, trends)
* Temporal dynamics (change from personal baseline)

### Behavioral Features

* Deviations from individual historical baselines
* Variance and consistency metrics

**Key idea**: Compare individuals to themselves over time, not to population averages.

---

## 8. Machine Learning Models

### Individual Models

* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Random Forest / Improved Random Forest (IRF)
* LightGBM

### Ensemble Strategy

* Soft voting across models
* Emphasis on recall (sensitivity)

### Design Bias

* Class weighting to penalize false negatives
* Lower decision thresholds to encode pessimism

---

## 9. Risk Scoring and Pessimistic Design

### Risk Bands

Model outputs are mapped to interpretable categories:

* Low Risk
* Might Be At Risk
* At Risk
* High Risk

### Pessimistic Logic

* Prioritize recall over precision
* Use temporal smoothing (rolling windows)
* Max-risk dominance across modalities

A single anomalous day does not trigger alerts; persistent trends do.

---

## 10. Confidence Estimation

Confidence reflects **reliability**, not certainty.

Factors contributing to confidence:

* Agreement among ensemble models
* Stability of risk over time
* Availability of input modalities

Outputs include:

* Risk band
* Confidence score
* Data coverage indicator

---

## 11. Evaluation Metrics

Primary metrics:

* Recall (Sensitivity)
* ROC-AUC
* Confusion Matrix (per class)

Accuracy is treated as secondary due to class imbalance and screening focus.

---

## 12. Ethics and Privacy Considerations

* No clinical diagnosis claims
* Explicit user consent required
* No raw audio storage; features only
* Risk alerts framed as **check-in suggestions**
* Designed as a caregiver support tool

---

## 13. Implementation Stack

* Python
* Scikit-learn
* LightGBM
* Librosa
* Pandas / NumPy
* Conda environment

### Inference Deployment

* REST API using Flask or FastAPI
* Model hosted on cloud/local server

### Visualization

* Streamlit or lightweight web dashboard

---

## 14. Research Plan and Timeline

### Phase 1: Literature Review

* Study voice and behavioral markers
* Analyze existing datasets and methodologies

### Phase 2: Data Preparation

* Dataset selection
* Feature extraction pipelines
* Input schema finalization

### Phase 3: Modeling

* Train individual models
* Build ensemble
* Tune pessimistic thresholds

### Phase 4: Validation and Analysis

* Evaluate metrics
* Feature importance analysis
* Ablation studies

### Phase 5: Presentation

* Dashboard creation
* System documentation
* Ethical impact discussion

---

## 15. Limitations and Future Work

### Current Limitations

* No real-time smartphone sensing
* Limited dataset diversity
* Not a clinical diagnostic tool

### Future Extensions

* Full mobile app integration
* Real-time passive sensing
* Longitudinal personalization
* Clinical collaboration and validation

---

## 16. Conclusion

This project presents a feasible, ethical, and forward-compatible approach to early mental health risk detection using voice and behavioral signals. By prioritizing pessimistic screening, interpretability, and modular design, the system lays a strong foundation for future real-world deployment while delivering meaningful academic value in the current phase.
