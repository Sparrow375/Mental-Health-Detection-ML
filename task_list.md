# Lumen. Release Build Revamp Task List

## Sprint: Lumen Closed-Loop Revamp

### Phase 1: Closed-Loop Data Schema & Model Refinement
- [ ] **T70**: Design `ObservationEntity` Room database table and SharedPreferences schema for storing daily observations, feedback states, and baseline confidence indices.
- [ ] **T71**: Update local Python `detector.py` and `l1_scorer.py` to adapt feature weights and adjust variance thresholds based on user confirmation/correction feedback.
- [ ] **T72**: Implement the Adaptive Gate logic to output observational pattern discoveries during onboarding and steady-state confirmations on calm days.

### Phase 2: Home Screen Hero Redesign
- [ ] **T73**: Redesign the Home screen UI to present a single prominent frosted-glass **Hero Observation Card** replacing the co-equal card grid.
- [ ] **T74**: Implement the inline Interpret Action Chips (`That Tracks`, `Not Quite`, `Tell me more`) with smooth transition animations and haptic feedback.
- [ ] **T75**: Build the inline Correction Micro-Flow (expanding structured alternate checkboxes when "Not Quite" is selected).
- [ ] **T76**: Collapse the secondary wellness utilities (Breathing Lotus, Digital Detox, Wind-Down, Habit Quests) into a clean, collapsible "Today's Tools" bottom shelf.

### Phase 3: Insights & Feed Redesign
- [ ] **T77**: Overhaul the Insights screen to lead with a ranked vertical feed of narrative discovery cards featuring "Validated by You" trust badges.
- [ ] **T78**: Relocate the circular Consistency Gauge and 6-Axis Radar/Spider chart into a collapsible "Explore Metrics & Baselines" panel.

### Phase 4: Contextual Check-In
- [ ] **T79**: Remove the static, isolated daily Check-In tab, replacing it with a minimal journal writing pad. Route daily mood/stress sliders to trigger contextually inside the Home screen's observation card.

### Phase 5: UI/UX Visual Polish
- [ ] **T80**: Remove all raw emojis and replace them with custom-styled vector icons. Restyle canvas breathing animations and gauges with thin, elegant glowing lines and modern sans-serif typography.
