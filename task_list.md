# Lumen Pivot Onboarding Refinement Task List

- [x] Task 1: Refactor Consistency Scaling Sliders in Onboarding
  - Convert 1-5 numerical scales to descriptive qualitative options (e.g., "Not Very Much" to "Very Much").
  - Update `LifestyleSlider` component to show qualitative text instead of numerical value.
- [x] Task 2: Rephrase Onboarding Prompts
  - Sanitize user-facing texts to avoid technical words like "model weights calibration", "Digital DNA calibration", or "calibration".
- [x] Task 3: Fix Permission Enabling Lag
  - Inspect permission clicking handling, especially accessibility and usage stats, to identify and resolve UI lag by moving disclosure dialogs out of the scrollable `LazyColumn`.
- [x] Task 4: Implement Daily & Monthly Check-in Notification Triggers
  - Register periodic wellness notification checks inside `MonitoringService`'s runTick.
  - Send daily check-in prompt and monthly check-in reminder when permitted by user.
- [x] Task 5: Shorten Onboarding PHQ-9 / GAD-7 Assessment
  - Reduce the onboarding screening length to a concise format (PHQ-2 and GAD-2) and scale scores.
