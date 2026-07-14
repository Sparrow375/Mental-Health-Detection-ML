with open("f:/Avaneesh/projects/MH detector/Mental-Health-Detection-ML/MHealth - Copy/app/src/release/java/com/example/mhealth/MainActivity.kt", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "fun generateBehavioralSummary" in line:
        print(f"generateBehavioralSummary: line {idx + 1}")
    if "fun MindfulBreathingCard" in line:
        print(f"MindfulBreathingCard: line {idx + 1}")
    if "fun TelemetrySnapshotCard" in line:
        print(f"TelemetrySnapshotCard: line {idx + 1}")
    if "fun HabitQuestsSection" in line:
        print(f"HabitQuestsSection: line {idx + 1}")
    if "fun WindDownCompanionCard" in line:
        print(f"WindDownCompanionCard: line {idx + 1}")
    if "fun DigitalDetoxCard" in line:
        print(f"DigitalDetoxCard: line {idx + 1}")
    if "fun InsightsScreen" in line:
        print(f"InsightsScreen: line {idx + 1}")
    if "fun CheckInScreen" in line:
        print(f"CheckInScreen: line {idx + 1}")
    if "fun SettingsScreen" in line:
        print(f"SettingsScreen: line {idx + 1}")
