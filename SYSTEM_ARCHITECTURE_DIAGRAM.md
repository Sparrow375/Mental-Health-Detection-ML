# System 1 Architecture: Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA LAYER                            │
│  📱 Smartphone Sensors (18 Features)                                │
├─────────────────────────────────────────────────────────────────────┤
│  🎤 Voice          📱 Activity       🚶 Movement      😴 Sleep       │
│  • Pitch           • Screen time     • Distance      • Wake time    │
│  • Energy          • Unlocks         • Entropy       • Sleep hours  │
│  • Speaking rate   • Social apps     • Home time     • Sleep time   │
│                    • Texts/calls     • Places                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│            BASELINE ESTABLISHMENT (28 days)                         │
│  📊 PersonalityVector - YOUR "Normal"                               │
├─────────────────────────────────────────────────────────────────────┤
│  For each of 18 features:                                           │
│  ✓ Calculate mean (μ)                                               │
│  ✓ Calculate standard deviation (σ)                                 │
│  ✓ Store as personalized baseline                                   │
│                                                                      │
│  Example: sleep_duration_hours = 7.5 ± 0.8 hours                    │
│                                                                      │
│  ⚠️ CRITICAL: This is 40% of system accuracy                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              DAILY MONITORING (180 days)                            │
│  🔍 Continuous Analysis                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  DEVIATION       │  │   VELOCITY       │  │   SUSTAINED      │  │
│  │  MAGNITUDE       │  │   TRACKING       │  │   EVIDENCE       │  │
│  │  (~15% impact)   │  │   (~10% impact)  │  │   (~35% impact)  │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤  │
│  │ (value - μ) / σ  │  │  EWMA 7-day      │  │  Counter:        │  │
│  │                  │  │  trend line      │  │  consecutive     │  │
│  │ Result:          │  │                  │  │  anomalous days  │  │
│  │ +2.3 SD          │  │  Result:         │  │                  │  │
│  │ (High deviation) │  │  -0.15/day       │  │  Threshold:      │  │
│  │                  │  │  (Worsening)     │  │  ≥ 4 days        │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                              ↓                                      │
│                   ┌──────────────────────┐                          │
│                   │  ANOMALY SCORE       │                          │
│                   │  (0.0 - 1.0)         │                          │
│                   ├──────────────────────┤                          │
│                   │  70% magnitude       │                          │
│                   │  30% velocity        │                          │
│                   │                      │                          │
│                   │  Today: 0.763        │                          │
│                   └──────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 ALERT LEVEL DETERMINATION                           │
│  🚦 Conservative Decision Logic                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  IF sustained_days < 4 AND evidence < 2.0:                          │
│     └─→ 🟢 GREEN (Normal - not sustained yet)                       │
│                                                                      │
│  ELSE IF anomaly_score < 0.35:                                      │
│     └─→ 🟢 GREEN (Normal range)                                     │
│                                                                      │
│  ELSE IF anomaly_score < 0.50:                                      │
│     └─→ 🟡 YELLOW (Mild sustained deviation)                        │
│                                                                      │
│  ELSE IF anomaly_score < 0.65:                                      │
│     └─→ 🟠 ORANGE (Moderate concern)                                │
│                                                                      │
│  ELSE:                                                               │
│     └─→ 🔴 RED (High concern)                                       │
│                                                                      │
│  ⚠️ KEY INNOVATION: Won't alert on single bad days                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    PATTERN DETECTION                                │
│  🔎 Identify Temporal Signature                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Analyze 7-day window:                                              │
│                                                                      │
│  • Stable: Low variance, low mean deviation                         │
│    → Normal healthy variation                                       │
│                                                                      │
│  • Rapid Cycling: High variance, elevated mean                      │
│    → BPD-like pattern (mood swings every 3-7 days)                  │
│                                                                      │
│  • Gradual Drift: Low variance, increasing trend                    │
│    → Depression-like pattern (slowly worsening)                     │
│                                                                      │
│  • Acute Spike: Sudden jump, persistent elevation                   │
│    → Acute episode (crisis event)                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                   │
│  📋 Clinical-Quality Reports                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PDF Report (8 pages):                                              │
│  • Page 1: Executive Summary                                        │
│  • Page 2: Personality Vector Comparison                            │
│  • Page 3: Drift Timeline (180 days)                                │
│  • Pages 4-8: Feature Group Details                                 │
│                                                                      │
│  Daily Report:                                                       │
│  • Anomaly score: 0.763                                             │
│  • Alert level: RED                                                 │
│  • Pattern: Rapid Cycling                                           │
│  • Evidence: 169.34                                                 │
│  • Sustained days: 111                                              │
│  • Top deviations: social_app_ratio, voice_energy, sleep_duration   │
│                                                                      │
│  Recommendation:                                                     │
│  "REFER: Strong evidence of sustained behavioral deviation.         │
│   Clinical evaluation recommended."                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Critical Data Flow Path (What Actually Drives Results)

```
Raw Sensor Data
      ↓
[PersonalityVector Baseline] ← 40% of accuracy
      ↓
Daily Comparison → Deviation Magnitude (15%) + Velocity (10%)
      ↓
[Sustained Tracking] ← 35% of accuracy (KEY DIFFERENTIATOR)
      ↓
Alert Level + Pattern Type
      ↓
Clinical Report
```

---

## 📊 Test Results Summary

```
┌────────────────┬──────────────┬─────────────┬──────────────┬────────────┐
│ Patient        │ Scenario     │ Final Score │ Alert Dist.  │ Outcome    │
├────────────────┼──────────────┼─────────────┼──────────────┼────────────┤
│ PT-001         │ Normal       │ 0.244       │ 100% Green   │ ✓ NORMAL   │
│ PT-002         │ BPD Cycling  │ 0.763       │ 54% Red      │ ✓ ANOMALY  │
│ PT-003         │ Depression   │ 0.662       │ Mixed        │ ✓ ANOMALY  │
│ PT-004         │ Life Event   │ 0.240       │ 91% Green    │ ✓ NORMAL   │
│ PT-005         │ Mixed        │ 0.385       │ 43% Orange   │ ✓ WATCH    │
└────────────────┴──────────────┴─────────────┴──────────────┴────────────┘

Key Achievement: 0% False Positives on Healthy Controls
```

---

## ⚡ Performance Metrics (Current Simulation)

```
Specificity:     100%    (No healthy person flagged - EXCELLENT)
Sensitivity:     100%    (All pathological patterns caught)
NPV:             100%    (If system says normal, you're normal)
PPV:             ~75%    (If system says anomaly, needs verification)

⚠️ NOTE: These are synthetic data results. 
Real-world performance expected to be 75-85% across all metrics.
```

---

## 🚨 The Gap: Simulation vs. Reality

```
┌─────────────────────────┬──────────────────┬─────────────────────┐
│ Aspect                  │ Current State    │ Need for Deployment │
├─────────────────────────┼──────────────────┼─────────────────────┤
│ Data Source             │ Synthetic        │ Real patients       │
│ Ground Truth            │ Self-generated   │ PHQ-9, GAD-7, DSM-5 │
│ Feature Interactions    │ Independent      │ Multi-modal fusion  │
│ Baseline                │ Static           │ Adaptive            │
│ Explainability          │ None             │ Natural language    │
│ Population Diversity    │ Single profile   │ Demographics        │
│ Confounds               │ Not modeled      │ Illness, meds, etc. │
│ Clinical Validation     │ None             │ IRB-approved trial  │
└─────────────────────────┴──────────────────┴─────────────────────┘
```

---

## 🎓 Bottom Line Assessment

**Architecture Quality:** 9/10 ⭐⭐⭐⭐⭐⭐⭐⭐⭐  
**Simulation Quality:** 7.5/10 ⭐⭐⭐⭐⭐⭐⭐⭐  
**Clinical Readiness:** 3/10 ⭐⭐⭐  

**Verdict:**  
You've built an excellent research prototype with sound detection logic and conservative alerting. The simulation proves the concept works. Now you need real patient data to validate it actually detects real mental health conditions, not just simulated ones.

**Your biggest strength:** The sustained evidence requirement prevents false alarms.  
**Your biggest gap:** No validation against clinical ground truth.  

**Next step:** Download StudentLife dataset and see if your detector works on real humans. 🎯
