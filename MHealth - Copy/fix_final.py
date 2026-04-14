import re

main_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\MainActivity.kt'
with open(main_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix Import
text = text.replace(
    'import com.example.mhealth.models.AnalysisResultEntity',
    'import com.example.mhealth.logic.db.AnalysisResultEntity'
)

# Fix HomeScreen missing variables
if 'val s1ProfileJson by DataRepository' not in text:
    target = r'(fun HomeScreen\(\)\s*\{)'
    replacement = r'\1\n    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()\n    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()\n    val analysisHistory by DataRepository.analysisHistory.collectAsState()'
    text = re.sub(target, replacement, text, count=1)

with open(main_file, 'w', encoding='utf-8') as f:
    f.write(text)

mon_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\ui\screens\MonitorScreen.kt'
with open(mon_file, 'r', encoding='utf-8') as f:
    text = f.read()

if 'import com.example.mhealth.alertColor' not in text:
    text = text.replace(
        'import com.example.mhealth.logic.DataRepository',
        'import com.example.mhealth.logic.DataRepository\nimport com.example.mhealth.alertColor'
    )

with open(mon_file, 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched successfully')
