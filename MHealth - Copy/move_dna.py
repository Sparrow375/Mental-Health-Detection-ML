import os

monitor_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\ui\screens\MonitorScreen.kt'
main_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\MainActivity.kt'

with open(monitor_file, 'r', encoding='utf-8') as f:
    monitor_content = f.read()

# 1. Fix null check mapping
monitor_content = monitor_content.replace(
    'val v = checkNotNull(vector); val b = checkNotNull(baseline)',
    'val v = vector ?: return@InfoCard; val b = baseline ?: return@InfoCard'
)

# 2. Extract items
start_marker = "        // ── System 1 DNA Profile Section"
item_end_marker = "        item { Spacer(Modifier.height(16.dp)) }"
start_idx = monitor_content.find(start_marker)
item_end_idx = monitor_content.find(item_end_marker)

items_str = ""
if start_idx != -1 and item_end_idx != -1:
    items_str = monitor_content[start_idx:item_end_idx]

# 3. Extract helpers
helper_start_marker = "// ── Helper composables for L2 visualization"
helper_idx = monitor_content.find(helper_start_marker)
helpers_str = ""
if helper_idx != -1:
    helpers_str = monitor_content[helper_idx:]

# 4. Remove from MonitorScreen
if start_idx != -1 and helper_idx != -1:
    new_monitor_content = monitor_content[:start_idx] + "\n" + monitor_content[item_end_idx:helper_idx].strip() + "\n}\n"
    with open(monitor_file, 'w', encoding='utf-8') as f:
        f.write(new_monitor_content)
    print("Updated MonitorScreen.kt")
else:
    print("Could not find markers in MonitorScreen")

# 5. Process MainActivity
with open(main_file, 'r', encoding='utf-8') as f:
    main_content = f.read()

# Add states to HomeScreen
if 'val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()' not in main_content:
    main_content = main_content.replace(
        'val vector by DataRepository.latestVector.collectAsState()',
        '''val vector by DataRepository.latestVector.collectAsState()
    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()
    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()
    val analysisHistory by DataRepository.analysisHistory.collectAsState()'''
    )

# Add items at the end of HomeScreen
system_stats_marker = "            // System stats row"
system_stats_idx = main_content.find(system_stats_marker)
# Find the end of HomeScreen blocks
insert_point = main_content.find("        }\n    }\n}", system_stats_idx)

if insert_point != -1 and items_str and items_str not in main_content:
    main_content = main_content[:insert_point] + "\n" + items_str + "\n" + main_content[insert_point:]

# Append helpers to the very end of MainActivity
if helpers_str and helpers_str not in main_content:
    main_content += "\n" + helpers_str

# Add imports if missing
imports_to_add = [
    "import com.example.mhealth.ui.screens.DnaProfileSection",
    "import com.example.mhealth.ui.charts.AnomalyScoreGauge",
    "import com.example.mhealth.ui.components.ScreenHeader",
    "import com.example.mhealth.ui.theme.MhealthAccentPurple",
    "import com.example.mhealth.ui.theme.MhealthTeal",
    "import com.example.mhealth.ui.theme.MhealthIndigo",
    "import com.example.mhealth.ui.theme.MhealthChartIndigo",
    "import com.example.mhealth.ui.theme.MhealthChartSlate",
    "import com.example.mhealth.ui.theme.TextMuted",
    "import androidx.compose.ui.draw.clip",
    "import androidx.compose.foundation.shape.CircleShape",
    "import androidx.compose.material.icons.filled.Shield",
    "import androidx.compose.foundation.layout.Arrangement"
]

for imp in imports_to_add:
    if imp not in main_content:
        # insert after first import
        imp_idx = main_content.find("import ")
        main_content = main_content[:imp_idx] + imp + "\n" + main_content[imp_idx:]

with open(main_file, 'w', encoding='utf-8') as f:
    f.write(main_content)
print("Updated MainActivity.kt")

