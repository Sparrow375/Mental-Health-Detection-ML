# Sprint: Lumen Closed-Loop Revamp Task List

## Phase 1: Data Schema & Python Model Integration (Closed-Loop Infrastructure)
- [x] **T70**: Design `ObservationEntity` Room table and SharedPreferences baseline confidence gate schema.
- [x] **T71**: Update Python `detector.py` to support baseline weight/variance tuning using confirmed/corrected user feedback.
- [x] **T72**: Implement the Adaptive Gate in Python to output simple pattern observations during cold-start and steady-state confirmations on quiet days.

## Phase 2: Home Screen Revamp (The Single Observation Hero)
- [x] **T73**: Redesign the Home screen to feature a single frosted-glass Hero Observation Card.
- [x] **T74**: Implement the inline Interpret Action Chips (`That Tracks`, `Not Quite`, `Tell me more`) with haptic feedback.
- [x] **T75**: Build the inline Correction Micro-Flow dialog providing structured alternate options.
- [x] **T76**: Collapse Breathing, Detox, Wind-Down, and Habits into a bottom "Today's Tools" tray.

## Phase 3: Insights Screen Revamp (The Discovery Feed)
- [x] **T77**: Pivot the Insights tab to lead with a ranked feed of narrative discovery cards with "Validated by You" badges.
- [x] **T78**: Move the Anomaly Gauge and 6-Axis Radar chart to a collapsible "Explore Metrics" drawer.

## Phase 4: Contextual Check-In Tab Revamp
- [x] **T79**: Replace the full static Check-In tab with a simplified minimalist journaling pad, routing daily sliders to trigger contextually off Home observations.

## Phase 5: UI/UX Premium Polish
- [x] **T80**: Overhaul the visual design system: replace emojis with vector icons, restyle canvas breathing lotus/gauge with elegant glows, and apply mature sans-serif typography.
