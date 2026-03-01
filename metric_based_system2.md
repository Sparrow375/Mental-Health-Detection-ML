# Metric-Based System 2: Clinical Prototype Matching

> **Status**: Design Proposal  
> **Date**: 2026-03-01  
> **Purpose**: A fully interpretable, metric-driven, training-data-free disorder classification engine for System 2.

---

## 1. The Core Problems This Solves

### Problem A: Contaminated Baseline
System 1's 28-day baseline blindly assumes the onboarding period represents a *healthy normal*. If the person is already depressed, manic, or schizophrenic during onboarding:
- Their disordered state gets recorded as "normal"
- Future recovery gets flagged as an anomaly
- The entire downstream pipeline is inverted

### Problem B: Directionless Anomaly Output
System 1 only outputs *"anomaly detected"* — it cannot tell whether the deviation is clinically concerning or a positive life change. Both a person withdrawing into depression and a person recovering from depression look like "deviated from baseline."

### Problem C: Closed-World Classifier Failure
A classifier trained on Depression + Schizophrenia will force-fit any patient (BPD, Bipolar, Anxiety, etc.) into one of those two categories. Confidently wrong is worse than honestly uncertain.

### Problem D: No Data for Many Disorders
BPD, Bipolar, Anxiety — good longitudinal passive-sensing datasets with clinical labels barely exist publicly. A pure ML approach is blocked without this data.

---

## 2. The Solution: Clinical Prototype Matching

Instead of training a classifier, we define a **prototype vector** for each disorder — the expected pattern of behavioral values grounded in clinical literature and DSM-5 criteria. Classification becomes a **geometric distance problem**, not a statistical learning problem.

```
User's behavioral pattern  ──►  Compare to all disorder prototypes  ──►  Nearest = likely pattern
```

No training data required. New disorders can be added by defining a new prototype vector.

---

## 3. Two Reference Frames — Critical Design Decision

> **This is the most important architectural choice in System 2.**

The system uses **two different reference frames** for two different purposes. Using only the personal baseline as a reference is insufficient because the baseline itself might be disordered — meaning a "deviated by 0" reading could still represent a disorder.

### Frame 1: Population-Anchored (Absolute Values)
**When used**: Onboarding period (Days 1–28) — baseline screening and initial state classification.  
**Reference point**: Healthy population averages from literature.  
**Why**: Allows us to detect if the user is *already* in a disordered state before their baseline is locked in. The personal baseline is NOT trusted yet.

### Frame 2: Personal-Baseline-Anchored (Z-Scores)
**When used**: Ongoing monitoring (Day 28+) — after the baseline has been verified clean.  
**Reference point**: The user's own verified-clean personal baseline.  
**Why**: Personalization — an introvert's "normal" for texts per day is their own baseline, not the population average. Once the baseline is confirmed healthy, deviations from it are meaningful.

```
Onboarding (Days 1-28): Frame 1 (population anchor)
         ↓ Baseline passes 3-Gate screening
Monitoring (Day 28+):  Frame 2 (personal anchor)
         ↓ Anomaly detected by S1
S2 Prototype Matching: Frame 2 disorder prototypes
```

If the baseline **fails** Gate screening → fall back to population synthetic baseline (Frame 1) for ongoing monitoring too.

---

## 4. The Disorder Prototype Library

### Frame 1: Population-Anchored Prototypes (Absolute Values)
Used during baseline screening to assess whether the user's onboarding behavior looks healthy or disordered.  
Values are derived from published passive sensing studies (see Section 11).

| Feature | Healthy Population | Depression | Schizophrenia | BPD | Bipolar (Depressive) | Bipolar (Manic) | Anxiety |
|---|---|---|---|---|---|---|---|
| `displacement` (km/day) | 4.5 | 1.8 | 1.2 | Variable | 1.5 | 7.0+ | 2.1 |
| `sleep_duration` (hrs) | 7.2 | 9.5 or 5.0 | Variable | Variable | 9.0 | 3.5 | 5.8 |
| `sleep_variance` (SD nights) | 0.4 | 0.6 | 2.5+ | 1.8 | 0.7 | 1.5 | 1.6 |
| `texts_per_day` | 35 | 11 | 5 | Very variable | 8 | 90+ | 20 |
| `social_ratio` | 0.28 | 0.10 | 0.05 | Very variable | 0.09 | 0.55+ | 0.18 |
| `response_time` (mins) | 8.2 | 28.0 | 35.0 | Erratic | 25.0 | 2.0 | Erratic |
| `screen_time` (hrs) | 4.0 | 5.5 | 3.5 | Variable | 5.0 | 7.5 | 5.2 |
| `location_diversity` | 4–6 places | 1–2 places | 1 place | Variable | 1–2 | 6+ | 2–3 |
| `app_diversity` (score) | 0.65 | 0.30 | 0.15 | Variable | 0.28 | 0.80 | 0.40 |
| `oscillation_freq` | Low | Low | Moderate-High | **Very High** | Low | — | Moderate |
| `velocity` | ~0 | Gradual ↓ | Variable | Rapid ± | Gradual ↓ | Sudden ↑ | Episodic ↑ |

> **Note**: "Variable" means the feature is diagnostically inconclusive for that disorder alone — other features must converge.

---

### Frame 2: Personal-Baseline-Anchored Prototypes (Z-Scores from Clean Baseline)
Used during ongoing monitoring (Day 28+), after the baseline has been verified clean.  
Values represent expected standard deviation **deviations from the user's own verified-clean baseline**.

#### 🟢 Healthy / Normal
```
All features:     ~0.0 SD    (behavior consistent with their verified healthy baseline)
oscillation_freq: LOW        (stable)
velocity:         ~0.0       (not trending in any direction)
co_deviation:     0–2        (at most 1-2 features slightly off)
```
> Why zeros here: The baseline was verified clean, so zero deviation from it genuinely means healthy.

#### 🔵 Depression
```
displacement:     -2.0 SD   (dramatically reduced movement)
social_ratio:     -1.8 SD   (social withdrawal)
texts_per_day:    -1.5 SD   (less communication)
calls_per_day:    -1.2 SD
sleep_duration:   +1.5 SD   (hypersomnia) OR -1.5 SD (insomnia — check which)
sleep_variance:   +0.5 SD   (shifted but consistent)
response_time:    +1.8 SD   (cognitive slowing)
screen_time:      +1.0 SD   (passive scrolling)
app_diversity:    -1.5 SD   (narrow, repetitive usage)
location_div:     -2.0 SD   (stays home)
oscillation_freq: LOW        (sustained directional drift, not cycling)
velocity:         NEGATIVE + GRADUAL (slow worsening over 3-8 weeks)
co_deviation:     5+         (multi-feature simultaneous collapse)
Temporal shape:   MONOTONIC DOWNWARD DRIFT
```

#### 🟣 Schizophrenia (Prodromal + Active)
```
displacement:     -1.5 SD   (restricted range)
social_ratio:     -2.5 SD   (severe withdrawal)
sleep_variance:   +3.0 SD   (severely chaotic schedule)
app_diversity:    -2.0 SD   (fixated, very narrow usage)
location_div:     -2.5 SD   (extremely restricted)
response_time:    +2.0 SD   (disorganized responsiveness)
oscillation_freq: MODERATE-HIGH  (erratic, irregular)
dev_variance:     +3.5 SD   (chaotic unpredictable swings)
co_deviation:     6+         (near-total behavioral disorganization)
Temporal shape:   DISORGANIZED CHAOS — neither consistent low nor regular cycling
```

#### 🟠 BPD (Borderline Personality Disorder)
```
social_ratio:     SWINGS ±3.0 SD  (periods of hypersocial AND withdrawal)
texts_per_day:    SWINGS ±3.0 SD  (high communication bursts then ghosting)
sleep_variance:   +2.5 SD         (significant disruption)
oscillation_freq: VERY HIGH        (rapid cycling EVERY 3-7 DAYS)
dev_variance:     +4.0 SD          (highest variance of all disorders)
velocity:         HIGH BOTH ± DIRECTIONS
Temporal shape:   RAPID REGULAR OSCILLATION between extremes
```

#### 🟡 Bipolar Disorder
```
Depressive phase:  (same as Depression prototype above)
Manic phase:
  displacement:    +2.0 SD   (hyperactive, more travel)
  social_ratio:    +2.5 SD   (hypersocial)
  sleep_duration:  -2.5 SD   (3-5 hrs — dramatically reduced)
  texts_per_day:   +3.0 SD   (impulsive high communication)
  screen_time:     +2.0 SD
  velocity:        SUDDEN HIGH POSITIVE (rapid flip from depressive phase)
oscillation_freq:  LOW-MODERATE  (long phases: weeks to months, not days)
Temporal shape:    EPISODIC PHASES — prolonged low then sudden flip to high
```

#### 🔴 Anxiety Disorder
```
sleep_duration:   -1.0 SD   (insomnia)
sleep_variance:   +2.0 SD   (fragmented)
location_div:     -1.5 SD   (avoidance behavior)
response_time:    HIGHLY VARIABLE  (erratic — sometimes hyper-fast, sometimes very slow)
unlock_freq:      +1.5 SD   (checking behavior, phone as safety object)
oscillation_freq: MODERATE   (episodic spikes, not constant)
dev_variance:     +2.5 SD   (volatile but not as extreme as BPD)
Temporal shape:   EPISODIC SPIKES — triggered by events → spike → partial recovery
```

---

## 5. Classification Engine: Distance Scoring

### Step 1: Choose Reference Frame
- If baseline **passed** Gate screening → use Frame 2 prototypes (personal z-scores from S1)
- If baseline **failed** Gate screening → use Frame 1 prototypes (compare raw values to population prototypes)

### Step 2: Build the Comparison Vector
**Frame 2 (usual case)**: Use S1's deviation vector directly:
```
U = [displacement_z, social_ratio_z, sleep_duration_z, sleep_variance_z,
     response_time_z, screen_time_z, app_diversity_z, location_div_z,
     oscillation_freq, dev_variance, velocity, co_deviation_count]
```

**Frame 1 (contaminated baseline fallback)**: Use raw 28-day average values normalized against population means.

### Step 3: Compute Match Score Against Each Prototype

**Cosine Similarity** (captures shape — which features are high/low relative to each other):
```
cos_sim(U, P_d) = (U · P_d) / (|U| × |P_d|)
Range: -1 to 1.  Closer to 1 = more similar shape.
```

**Weighted Euclidean Distance** (captures magnitude of match):
```
dist(U, P_d) = sqrt( Σ w_i × (U_i - P_d_i)² )
Lower = closer match.
Feature weights w_i can prioritize more diagnostically reliable features.
```

**Combined Match Score**:
```
match_score(d) = 0.6 × cos_sim(U, P_d) + 0.4 × (1 / (1 + dist(U, P_d)))
```

### Step 4: Rank and Apply Confidence Thresholds
```
match_score(Normal):        0.91
match_score(Depression):    0.37
match_score(Schizophrenia): 0.22
match_score(BPD):           0.18
...

IF max_score ≥ 0.75 → Output classification with HIGH CONFIDENCE
IF max_score 0.55–0.75 → Output "Possible [X]" with LOW CONFIDENCE — monitor
IF max_score < 0.55 → Output "UNCLASSIFIED — escalate for clinical review"
```

---

## 6. Temporal Shape Validation (Second-Pass Check)

After distance scoring gives a tentative classification, validate it against the **temporal shape** of the anomaly trajectory:

| Temporal Pattern Detected | Supports | Contradicts |
|---|---|---|
| Monotonic downward drift (4–8 weeks) | Depression | BPD, Bipolar |
| Chaotic high-variance, no direction | Schizophrenia | Depression |
| Regular oscillation every 3–10 days | BPD | Depression, Schizophrenia |
| Long depressive phase → sudden flip upward | Bipolar | Depression (alone) |
| Brief spike → gradual recovery (2–3 weeks) | Anxiety / Life Event | Chronic disorders |

```
if temporal_shape SUPPORTS classification:
    confidence × 1.2   (boost)
elif temporal_shape CONTRADICTS classification:
    confidence × 0.6   (downgrade, add uncertainty flag)
    flag for second-opinion review
```

---

## 7. Baseline Contamination Screening (3-Gate Filter)

Run ALL THREE gates during onboarding (Days 1–28) using **Frame 1** (population-anchored values). If any gate fails, the baseline is flagged as potentially contaminated.

---

### Gate 1: Population Anchor Check (Day 7)
After 7 days, compare the emerging raw behavioral values against healthy population norms for the user's demographic:

```
For each feature f:
    z_pop = (user_value_f - population_mean_f) / population_std_f

IF |z_pop| > 2.5 for THREE OR MORE features simultaneously:
    → FLAG: "Possible pre-existing condition — do not lock baseline yet"
    → Extend monitoring, prompt self-report (PHQ-9 / GAD-7)
```

---

### Gate 2: Internal Stability Check (Day 14–21)
A healthy person's baseline should be **stable** week over week. Someone actively cycling or in an acute episode will show high within-baseline variance:

```
For each feature f:
    week1_mean = mean(days 1–7)
    week2_mean = mean(days 8–14)
    week3_mean = mean(days 15–21)

    baseline_drift_f = std([week1_mean, week2_mean, week3_mean])

IF baseline_drift_f > 1.5 × population_expected_drift for 3+ features:
    → FLAG: "Unstable baseline — possible cycling disorder (BPD/Bipolar) or active episode"
```

---

### Gate 3: Prototype Proximity Check (Day 28)
Before locking the baseline, run the Frame 1 prototype matching on the 28-day raw behavioral average:

```
Run match_score(user_28day_raw_values, Frame1_prototypes)

IF top match is NOT "Healthy Normal" AND confidence > 0.65:
    → CONTAMINATED BASELINE CONFIRMED
    → Do NOT use personal baseline as P₀
    → Fall back to population synthetic baseline for Frame 2 reference
    → Optionally prompt: "We want to make sure we understand your patterns well —
       would you like to take a quick check-in questionnaire?"
```

> ⚠️ Never tell the user their baseline was "contaminated." Frame it as: *"We're still learning your patterns and want to make sure we get it right."*

---

### What Happens on Contamination Detection?

| Gate(s) That Fire | Action |
|---|---|
| Gate 1 only | Extend baseline to 56 days, monitor carefully |
| Gate 2 only | Flag possible cycling disorder, continue monitoring |
| Gate 3 fires | Replace personal baseline with synthetic population normal |
| Gate 1 + Gate 3 | Replace baseline + prompt PHQ-9/GAD-7 self-report |
| All three | Treat as likely pre-existing condition, refer clinical review |

---

## 8. Full System 2 Pipeline

```
[System 1: Anomaly Detected]
         │
         ▼
[Stage 0: Life Event Filter]
  Is the anomaly confined to 1–2 features AND resolved within 10 days?
  → YES: DISMISS (likely situational, not a disorder)
  → NO: proceed
         │
         ▼
[Stage 1: Select Reference Frame]
  Did baseline pass all 3 Gates?
  → YES: Use Frame 2 (personal z-scores)
  → NO:  Use Frame 1 (population-anchored raw values)
         │
         ▼
[Stage 2: Distance Scoring vs All Prototypes]
  Compute match scores against all disorder prototypes
  → High confidence (≥0.75): tentative classification
  → Low confidence (<0.55): mark UNCERTAIN
         │
         ▼
[Stage 3: Temporal Shape Validation]
  Does the trajectory shape confirm or contradict the classification?
  → Adjust confidence up or down
         │
         ▼
[Output]
  "Consistent with [Disorder] pattern" — Confidence: X%
  Top 3 deviating features driving the result (explainability)
  OR: "Uncertain — does not match known profiles, escalate"
  OR: "Likely life event or situational stress"
```

---

## 9. Radar Chart Visualization

Plot the user's behavioral profile against disorder prototypes on a radar/spider chart — one axis per behavioral dimension. The shape that most overlaps with the user's profile indicates the closest disorder match.

```
          Social Ratio
               │
  Location ────┼──── Sleep Duration
  Diversity    │
               │
  App       ───┼─── Response
  Diversity    │     Time
               │
          Displacement
```

- **Gray shaded area**: Healthy population prototype (Frame 1) or clean personal baseline (Frame 2)
- **Colored dashed outline**: Disorder prototype being compared
- **Solid filled area**: User's actual profile
- Clinicians can see immediately which prototype *shape* the user's profile resembles — no ML interpretation needed

---

## 10. Why This Beats a Pure ML Classifier

| Aspect | ML Classifier | Prototype Matching (This Approach) |
|---|---|---|
| Training data needed | Large labeled dataset per disorder | None — clinical knowledge only |
| New disorders | Retrain entire model | Add one prototype vector |
| Interpretability | Black box | Every decision fully explainable |
| Confidence handling | Overconfident softmax | Natural distance threshold |
| Unknown disorders | Force-fits into known class | "Unclassified" output |
| Clinician adjustable | Cannot adjust without retraining | Edit prototype weights directly |
| Comorbidity | Single output | Multiple match scores shown |
| Contaminated baseline | No protection | Detected and corrected via 3-Gate filter |

Your labeled Depression + Schizophrenia data is used to **validate and calibrate** prototype weights — not to train a classifier. Compute actual mean z-scores from your dataset's high-PHQ vs. low-PHQ groups, compare against the prototype values, and adjust until they align.

---

## 11. Limitations and Mitigations

| Limitation | Mitigation |
|---|---|
| Prototype weights are hand-crafted | Calibrate using your StudentLife labeled data |
| Population norms need demographic stratification | Use StudentLife + published literature tables |
| High individual behavioral variation | Frame 2 personalizes once baseline is clean |
| Cannot detect disorders not in the library | "Unclassified" output = graceful failure, not silent wrong answer |
| No learning over time | Add lightweight feedback: clinician confirms → nudge prototype weights |

---

## 12. Source Literature for Prototype Values

- **Saeb et al. (2015)**: GPS + mobility in depressed students — JMIR Mental Health
- **Canzian & Musolesi (2015)**: Mobility traces → depression correlation — UbiComp
- **Barnett et al. (2018)**: Passive sensing in schizophrenia — npj Schizophrenia
- **Faurholt-Jepsen et al. (2015)**: Bipolar smartphone sensing — MONARCA study
- **Santangelo et al. (2014)**: BPD ecological momentary assessment
- **Boukhechba et al. (2018)**: Anxiety avoidance in passive sensing
- **StudentLife Dataset**: PHQ-9 + behavioral features, 48 students — Dartmouth

---

*Prototype values should be reviewed by a clinical collaborator (psychiatrist/psychologist) before deployment.*
