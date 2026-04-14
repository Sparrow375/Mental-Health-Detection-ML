import os

main_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\MainActivity.kt'
with open(main_file, 'r', encoding='utf-8') as f:
    text = f.read()

if 'import com.example.mhealth.models.AnalysisResultEntity' not in text:
    text = text.replace(
        'import com.example.mhealth.models.PersonalityVector',
        'import com.example.mhealth.models.PersonalityVector\nimport com.example.mhealth.models.AnalysisResultEntity'
    )

with open(main_file, 'w', encoding='utf-8') as f:
    f.write(text)


mon_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\ui\screens\MonitorScreen.kt'
with open(mon_file, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('val analysisResult by DataRepository.latestAnalysisResult', 'val latestResult by DataRepository.latestAnalysisResult')
text = text.replace('analysisResult?.let {', 'latestResult?.let {')
text = text.replace('analysisResult?.let{', 'latestResult?.let{')

with open(mon_file, 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched successfully')
