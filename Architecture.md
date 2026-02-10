## System Architecture Overview

### **LAYER 1: Data Collection (The Sensor)**
**Mobile app continuously monitors:**
- **Audio**: Voice pitch, tempo, volume, speech rate, pauses, stuttering
- **Activity**: Screen time, app usage patterns, typing speed, call/text frequency
- **Movement**: Location changes, step count, time spent at locations
- **Phone interaction**: Unlock frequency, notification response time
- **Sleep proxy**: Phone usage patterns at night, movement data
- **Optional self-report**: Quick daily mood check-ins (1-10 scale)

**Sampling strategy:**
- Passive data: Continuous collection
- Voice: Triggered by calls, voice notes, or opt-in recording windows
- Preprocessing at edge (on-device) for privacy

---

### **LAYER 2: Baseline Establishment (Weeks 1-4+)**
**Goal**: Build the "Personal Normal" vector

**Process:**
1. Collect all multimodal data for minimum 4-6 weeks
2. Screen for existing symptoms (brief questionnaire) to ensure clean baseline
3. Calculate baseline metrics:
   - **Voice**: Mean/SD of pitch, tempo, energy
   - **Activity**: Average phone usage, social interaction frequency
   - **Movement**: Typical location radius, daily displacement
   - **Temporal**: Standard wake/sleep times, daily routines

**Output**: 
- **Personality Vector (P₀)**: Multi-dimensional baseline profile
- **Variance bounds**: Normal fluctuation ranges for each feature

---

### **LAYER 3: Continuous Monitoring (Post-baseline)**
**Process:**
- Collect same data as baseline period
- Calculate current state vector (P_current) over sliding windows:
  - **Short window**: 24-48 hours (acute changes)
  - **Medium window**: 1 week (trend detection)
  - **Long window**: 2-4 weeks (persistent shifts)

**Feed both to:**
- System 1 (anomaly detection)
- Baseline adaptation logic

---

### **SYSTEM 1: Anomaly Detection Engine**

**Core function**: Detect deviations from P₀

**Calculates:**
1. **Deviation magnitude**: |P_current - P₀| for each feature
2. **Deviation velocity**: Rate of change over time
3. **Deviation frequency**: How often changes occur
4. **Recovery time**: Time between fluctuations returning to baseline
5. **Multi-feature correlation**: Are multiple domains changing together?

**Outputs:**
- **Anomaly score** (0-1): Overall deviation severity
- **Feature-specific flags**: Which aspects are deviating
- **Temporal pattern**: Rapid cycling vs gradual drift vs episodic spikes
- **Trend direction**: Moving away from or toward baseline

**Thresholds**:
- Mild: 1-2 SD from baseline
- Moderate: 2-3 SD
- Severe: >3 SD or persistent 2+ SD for >1 week

---

### **SYSTEM 2: Disorder Characterization (Classification Layer)**

**Inputs from System 1:**
- Anomaly scores
- Deviation patterns
- Affected feature domains
- Temporal dynamics

**Additional raw features:**
- Specific voice characteristics
- Social interaction patterns
- Circadian disruption metrics

**Pattern matching logic:**

| Disorder | Signature Pattern |
|----------|------------------|
| **BPD** | High deviation velocity + frequent rapid cycling + relationship/social volatility |
| **Depression** | Gradual drift downward + sustained low activity + social withdrawal + speech slowing |
| **Bipolar** | Episodic bidirectional swings + decreased sleep with increased activity (mania) |
| **Anxiety** | Episodic spikes + physiological markers (if available) + avoidance patterns |
| **Schizophrenia** | Speech disorganization + social withdrawal + circadian disruption + gradual onset |

**Model approach:**
- **Option A**: Multi-class classifier (Random Forest, XGBoost, Neural Net)
- **Option B**: Ensemble of binary classifiers per disorder
- **Option C**: Probabilistic graphical model (accounts for comorbidity)

**Outputs:**
- **Probability distribution** over disorders (not single diagnosis)
- **Confidence score**: How certain is the pattern match
- **Key contributing features**: What's driving the prediction
- **Trajectory**: Getting worse/stable/improving

---

### **LAYER 4: Baseline Adaptation (Dynamic Learning)**

**Handles legitimate life changes vs. pathological shifts**

**Logic:**
1. If deviation is **persistent** (>4-6 weeks) AND **gradual** AND **stabilizes at new level**:
   - Likely life change (new job, relationship, moved cities)
   - Gradually shift P₀ toward new stable state
   
2. If deviation is **fluctuating** OR **progressive deterioration** OR **acute**:
   - Likely pathological
   - Don't update baseline; flag for monitoring

**Safety**: Require manual confirmation or clinical validation before major baseline shifts

---

### **LAYER 5: Alert & Output System**

**Tiered alert system:**

**Green**: Within normal bounds
- No action

**Yellow**: Mild deviation detected
- In-app notification: "We've noticed some changes in your patterns. How are you feeling?"
- Log for tracking

**Orange**: Moderate deviation or concerning pattern
- Stronger notification
- Suggest self-care resources
- Optional: Share with designated trusted contact

**Red**: Severe or persistent deviation + high disorder probability
- Urgent notification
- Recommend professional consultation
- Provide crisis resources if needed

**Dashboard for user:**
- Trends over time (visualized)
- Which areas are changing
- Educational content about patterns
- NOT a diagnosis, framed as "pattern awareness"

---

### **Data Flow Summary**

```
Sensors (phone) 
    ↓
Data Collection Layer
    ↓
[Baseline Phase] → P₀ (Personality Vector)
    ↓
[Monitoring Phase] → P_current
    ↓
System 1 (Anomaly Detection)
    ├→ Deviation metrics
    └→ Temporal patterns
         ↓
System 2 (Disorder Characterization)
    ├→ Pattern matching
    └→ Probability scores
         ↓
Alert System + User Dashboard
         ↓
[Feedback loop] → Baseline Adaptation
```

---

### **Key Technical Considerations**

1. **Privacy**: All processing on-device where possible; encrypted cloud storage only for aggregated/anonymized data
2. **Battery**: Optimize collection to minimize drain
3. **Storage**: Efficient compression; rolling window storage
4. **Interpretability**: SHAP values or attention mechanisms to explain predictions
5. **Validation**: Need longitudinal clinical data to train System 2

This architecture separates concerns cleanly: collect → establish baseline → detect anomalies → characterize patterns → adapt baseline → alert. Each layer can be developed and validated independently.


