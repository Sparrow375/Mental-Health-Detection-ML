# System 2: 6-Phase Clinical Diagnostic Pipeline 

**Status**: IMPLEMENTED natively running on Edge via Chaquopy (`pipeline.py`).

---

## 1. The Core Problems This Solves

### A. Non-Directional Anomaly Outputs
System 1 is brilliant at identifying deviations, but blind to cause. To System 1, a patient radically improving their sleep schedule and a patient spiraling into depressive insomnia both trigger "Statistical Anomaly." System 2 bridges the gap between math and psychiatry.

### B. Closed-World Classifier Failure
Traditional ML classifiers completely fail when confronted with disorders they weren't trained on. If you rain an ML model on Depression + Schizophrenia, it will confidently force a BPD or Anxiety episode into one of those two boxes. System 2 solves this via geometric distance mapping. 

---

## 2. The 6-Phase Pipeline Orchestration

When System 1 triggers a sustained anomaly with a high evidence score, `engine.py` packages the behavioral deviations and sends them sequentially through System 2's `pipeline.py` triage line:

### Phase 1: Baseline Screener
**Objective:** Prevents "The Frozen Sick" paradox.
- Evaluates if the user was already severely depressed/schizophrenic during their initial 28-day calibration phase, meaning their "normal" is actually diseased.
- Fails the baseline and forces external population-norm anchoring if the patient fails the baseline health gate.

### Phase 2: Life Event Filter
**Objective:** Eliminates situational false positives.
- If GPS entropy drops to zero, and screen time hits 0, System 1 screams absolute panic. 
- The Life Event Filter checks contextual boundaries. Was this a sudden, isolated 4-day anomaly that self-resolved? It classifies it as "Going Camping/Off-Grid" or a "Flu Episode" and safely dismisses the alert.

### Phase 3: Distance Prototype Matcher
**Objective:** Training-data-free disorder classification. 
- We define a **prototype vector** for each disorder grounded in DSM-5 literature (e.g. Depression = lower movement, lower calls, higher sleep; Mania = high displacement, low sleep).
- The engine calculates a **Weighted Euclidean Distance** and **Cosine Similarity** between the user's current System 1 Z-Score array and every known psychiatric prototype. 
- The mathematically closest geometry assigns the likely structural disorder (e.g. "88% shape match with Bipolar Depressive phase").

### Phase 4: Clinical Guardrails
**Objective:** Override pure geometric ML matches with strict medical reality.
- *Psychosis Cluster Rule:* Mathematically, certain bipolar signatures overlap with Schizophrenia. The Guardrail enforces a hard rule: if extreme social withdrawal triggers *simultaneously* with catastrophic location entropy collapse, it forces a Schizophrenia-Type flag bypass.
- *Withdrawal Cluster Rule:* Solidifies major depressive classifications if digital communication effectively ceases.

### Phase 5: Temporal Validator
**Objective:** Confirms the duration-shape matches the disease classification.
- BPD exhibits *Rapid Regular Oscillation* (swings every 3-7 days). 
- Depression exhibits *Monotonic Downward Drift* (degrades stably over weeks).
- The Validator enforces logic: If the Distance Matcher chose "Depression", but the timeline shows manic flickering every 4 days, it heavily downgrades the statistical confidence and demands manual review.

### Phase 6: Explainability Engine
**Objective:** De-construct the math for the Admin Dashboard.
- Evaluates the winning arrays and dynamically produces human-readable context sentences.
- Example: *"High Confidence Pattern Match (Depression). Triggered primarily by a -2.1 SD collapse in daily step displacement and a +1.8 SD spike in hypersomniac sleep duration."*

---

## 3. The 2-Frame Reference Philosophy

System 2 dynamically switches its core mathematical grounding depending on the outcome of Phase 1:

**Frame 1: Population-Anchored (Absolute Values)** 
Used if the user has a contaminated baseline. We compare raw metrics (e.g., 2 hours of sleep) directly against global CDC literature averages.

**Frame 2: Personal-Anchored (Z-Scores)** 
Used when the baseline is verified healthy. We rely purely on internal variances. If an introvert has low social contact, it yields a $0.0z$ deviation. Only true internal shifts are flagged.
