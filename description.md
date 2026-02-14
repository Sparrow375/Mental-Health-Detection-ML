# Mental Health Anomaly Detection System - System 1

## 📋 Overview

This is an **Improved Anomaly Detection System** designed to detect potential mental health conditions (such as depression, bipolar disorder, etc.) by analyzing behavioral patterns from smartphone sensor data over time.

The key philosophy: **Don't panic on a single bad day** - the system only flags issues after accumulating sufficient evidence of **sustained behavioral changes** over multiple days.

---

## 🎯 What Does This System Do?

The system monitors a person's daily behavior across multiple dimensions and compares it to their personal "normal" baseline. It looks for:

- **Sustained deviations** from normal patterns (not just one-off bad days)
- **Gradual drifts** that might indicate developing mental health issues
- **Rapid cycling patterns** that could suggest conditions like bipolar disorder
- **Evidence accumulation** over time to avoid false alarms

---

## 🏗️ System Architecture

The system consists of 4 main components:

### 1. **Data Structures** (Lines 34-133)
Defines how information is stored:

- **PersonalityVector**: Your unique "fingerprint" of normal behavior
  - Voice features (pitch, energy, speaking rate)
  - Activity features (screen time, app usage, social interactions)
  - Movement features (locations visited, time at home)
  - Sleep patterns (wake time, sleep duration)

- **DailyReport**: What happened each day
  - Anomaly score for the day
  - Alert level (green/yellow/orange/red)
  - Which features were unusual
  - Evidence accumulated so far

- **AnomalyReport**: Real-time detection results
  - Current deviation scores
  - Rate of change (getting worse/better?)
  - Pattern type identified

- **FinalPrediction**: Summary after monitoring period
  - Overall status (anomaly detected or normal)
  - Confidence level
  - Clinical recommendation

### 2. **Synthetic Data Generator** (Lines 138-350)
Creates realistic fake data for testing purposes:

- **Baseline Generation**: Simulates 28 days of "normal" behavior to establish what's typical for a person
- **Scenario Simulation**: Generates 180 days (6 months) of monitoring data with different patterns:
  - `normal`: Healthy person with natural day-to-day variations
  - `bpd_rapid_cycling`: Bipolar pattern with rapid mood swings every 3-7 days
  - `anomaly_gradual_depression`: Slowly worsening depression over 6 months
  - `normal_life_event`: Healthy person experiencing temporary stress (e.g., bereavement)
  - `mixed_signals`: Complex, unclear patterns

### 3. **Anomaly Detector** (Lines 356-658)
The brain of the system - analyzes behavior and detects problems:

**Key Features:**
- Tracks **14 days of history** to understand trends
- Requires **4+ consecutive days** of deviation before raising alerts
- Accumulates **evidence score** that grows with sustained patterns
- Considers both **magnitude** (how far from normal) and **velocity** (rate of change)

**Main Methods:**
- `calculate_deviation_magnitude()`: Measures how many standard deviations away from baseline
- `calculate_deviation_velocity()`: Detects if things are getting worse/better over time
- `detect_pattern_type()`: Identifies if pattern is stable, cycling, drifting, etc.
- `calculate_anomaly_score()`: Overall score from 0-1 (0=normal, 1=severe)
- `update_sustained_tracking()`: Keeps track of how many consecutive "bad" days
- `determine_alert_level()`: Decides on green/yellow/orange/red status

### 4. **Report Generator & Visualization** (Lines 672-1010)
Creates detailed PDF reports and charts showing:

- **Executive Summary**: Overall status, recommendation, alert distribution
- **Personality Vector Comparison**: How current behavior differs from baseline
- **Drift Timeline**: How each feature changed over 180 days
- **Feature Details**: Deep dive into each category (voice, activity, sleep, etc.)

---

## 🔄 System Flow

Here's how the system works step-by-step:

### Phase 1: Baseline Establishment (28 days)
```
1. Collect data for 28 days
2. Calculate average values for all 18 features
3. Calculate standard deviations (natural variation)
4. Create PersonalityVector (your unique baseline)
```

### Phase 2: Monitoring (180 days)
For each day:
```
1. Collect current day's data
2. Compare to baseline → Calculate deviations
3. Track trend over past 7 days → Calculate velocity
4. Compute anomaly score (0-1)
5. Update sustained deviation counter
   ├─ If anomaly_score > 0.35: counter++, accumulate evidence
   └─ If anomaly_score < 0.35: counter--, decay evidence
6. Determine alert level
   ├─ Green: Normal or not sustained yet
   ├─ Yellow: Mild sustained deviation
   ├─ Orange: Moderate sustained deviation
   └─ Red: Severe sustained deviation
7. Store daily report
```

### Phase 3: Pattern Detection
```
1. Analyze 7-day windows to identify pattern type:
   ├─ Stable: Consistent behavior
   ├─ Rapid Cycling: High variability (bipolar-like)
   ├─ Gradual Drift: Steadily worsening (depression-like)
   └─ Mixed Pattern: Unclear signals
```

### Phase 4: Final Analysis
```
1. Review entire 180-day period
2. Check if sustained anomaly detected:
   ├─ Sustained days >= 4 OR
   └─ Evidence accumulated >= 2.0
3. Calculate confidence (based on data length)
4. Generate recommendation:
   ├─ REFER: Clinical evaluation needed
   ├─ MONITOR: Continue watching closely
   ├─ WATCH: Some concern, extend monitoring
   └─ NORMAL: No action needed
5. Create detailed PDF report & visualizations
```

---

## 📊 Understanding the Results

### Alert Levels
- 🟢 **Green**: Everything normal or not sustained enough to worry
- 🟡 **Yellow**: Mild sustained deviation detected
- 🟠 **Orange**: Moderate concern - sustained and significant deviation
- 🔴 **Red**: High concern - severe sustained deviation

### Anomaly Score
- **0.0 - 0.35**: Normal range
- **0.35 - 0.50**: Mild deviation
- **0.50 - 0.65**: Moderate deviation
- **0.65+**: Severe deviation

### Evidence Accumulation
- Grows when anomaly scores are sustained over multiple days
- **< 1.5**: Insufficient evidence
- **1.5 - 2.0**: Some evidence, continue monitoring
- **≥ 2.0**: Strong evidence threshold - triggers alerts

### Sustained Deviation Days
- Consecutive days with anomaly score > 0.35
- **< 4 days**: Not sustained - might be normal variation
- **≥ 4 days**: Sustained pattern - raises concern level

---

## 🧮 Key Algorithms Explained

### 1. Deviation Calculation
```
For each feature:
deviation = (current_value - baseline_mean) / baseline_std
```
*Example:* If baseline sleep is 7.5 ± 0.8 hours, and today is 10 hours:
- Deviation = (10 - 7.5) / 0.8 = **+3.1 standard deviations** (unusually high)

### 2. Velocity Calculation
```
1. Take last 7 days of feature values
2. Fit a line through them (linear regression)
3. velocity = slope / baseline_mean
```
*Purpose:* Detects if behavior is trending worse (negative slope) or improving (positive slope)

### 3. Anomaly Score
```
magnitude_score = average(|all deviations|) / 3.0  [capped at 1.0]
velocity_score = average(|all velocities|) * 10    [capped at 1.0]
anomaly_score = 0.7 * magnitude_score + 0.3 * velocity_score
```
*Weighted:* 70% based on current deviation, 30% based on trend direction

### 4. Evidence Accumulation
```
If anomaly_score > 0.35:
    evidence += anomaly_score * (1 + sustained_days * 0.1)
    # Evidence grows faster the longer deviation persists
Else:
    evidence *= 0.92  # Slowly decay if normal day occurs
```

---

## 📁 Generated Outputs

When you run the system, it creates:

### 1. PDF Reports (`report_*.pdf`)
Clinical-quality reports with:
- Executive summary page
- Personality vector comparison
- Timeline of all features over 6 months
- Detailed feature group pages

### 2. PNG Visualizations (`analysis_*.png`)
Summary charts showing:
- Key feature timelines
- Anomaly score evolution
- Evidence accumulation
- Alert distribution

### 3. Text Files
- `daily_report_*.txt`: Day-by-day summary
- `comparison_summary.txt`: Compare all scenarios side-by-side

---

## 🎭 Test Scenarios Explained

The system tests 5 different patient profiles:

| Scenario | Patient ID | What It Simulates | Expected Result |
|----------|-----------|-------------------|-----------------|
| `normal` | PT-001 | Healthy person, natural variation | ✅ No anomaly |
| `bpd_rapid_cycling` | PT-002 | Bipolar disorder with 3-7 day mood swings | ⚠️ Anomaly detected (cycling pattern) |
| `anomaly_gradual_depression` | PT-003 | Depression developing over 6 months | ⚠️ Anomaly detected (gradual drift) |
| `normal_life_event` | PT-004 | Healthy person with temporary stressor around day 70-100 | ✅ Should NOT flag as anomaly* |
| `mixed_signals` | PT-005 | Complex, unclear behavioral changes | ⚠️ May or may not flag |

*The system should distinguish temporary life stress from clinical conditions

---

## 🔑 Key Innovation

**Conservative Detection Philosophy:**
Traditional systems might flag a person after just 1-2 bad days. This system intentionally waits for:
- At least 4 consecutive days of deviation
- OR evidence score accumulating to 2.0+

This prevents false alarms from:
- Having a cold/flu
- One stressful week at work
- Temporary life events
- Normal mood fluctuations

While still catching:
- Sustained depression
- Bipolar cycling patterns
- Gradual mental health deterioration

---

## 🚀 How to Run

```python
python system1.py
```

This will:
1. Run all 5 test scenarios (takes a few minutes)
2. Generate PDF reports for each patient
3. Create visualization charts
4. Print summary to console
5. Save comparison table

---

## 📈 Features Monitored (18 Total)

### Voice Features (4)
- 🎤 Pitch mean & variation
- 📢 Energy level
- 🗣️ Speaking rate

### Digital Activity (3)
- 📱 Screen time
- 🔓 Phone unlocks
- 👥 Social app usage

### Social Connection (4)
- ☎️ Calls per day
- 💬 Texts per day
- 👤 Unique contacts
- ⏱️ Response time

### Movement (4)
- 🚶 Daily distance traveled
- 📍 Location variety (entropy)
- 🏠 Time spent at home
- 🗺️ Places visited

### Sleep & Circadian (3)
- ⏰ Wake time
- 🌙 Sleep time
- 😴 Sleep duration

---

## 💡 Real-World Application

In a real deployment, instead of synthetic data, the system would connect to:
- **Smartphone sensors**: GPS, accelerometer, screen events
- **App usage APIs**: Social media time, messaging patterns
- **Voice analysis**: From phone calls (with consent)
- **Wearable devices**: Sleep tracking, heart rate

The same detection logic would apply, creating a continuous mental health monitoring system that could alert clinicians to concerning patterns before a crisis occurs.

---

## ⚠️ Important Notes

1. **Not a Diagnostic Tool**: This system detects behavioral anomalies, not diagnoses. Clinical evaluation is always required.

2. **Privacy-First**: In real use, all data should be encrypted and user must give informed consent.

3. **Personalized Baselines**: The system learns YOUR normal - what's concerning for you might be normal for someone else.

4. **Requires Calibration**: The 4-day threshold and 2.0 evidence threshold are tunable parameters that should be validated with real clinical data.

---

## 📞 Summary

This system is a sophisticated early warning system for mental health issues. It's like a smoke detector for your behavioral patterns - designed to catch concerning changes early while avoiding false alarms from everyday stress and variation.

**Key Strengths:**
✅ Personalized to individual baselines  
✅ Requires sustained evidence before alerting  
✅ Tracks multiple behavioral dimensions  
✅ Generates clinical-quality reports  
✅ Distinguishes patterns (cycling vs. drift vs. stable)  

**Use Case:** 
Continuous monitoring of at-risk populations, early intervention programs, or post-treatment relapse prevention.
