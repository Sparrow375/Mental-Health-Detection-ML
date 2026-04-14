import re

main_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\MainActivity.kt'
with open(main_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove the incorrectly placed variables (anywhere they exist)
text = text.replace('    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()\n', '')
text = text.replace('    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()\n', '')
text = text.replace('    val analysisHistory by DataRepository.analysisHistory.collectAsState()\n', '')

# Insert at the right place
target = '    val context = LocalContext.current'
replacement = '    val context = LocalContext.current\n    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()\n    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()\n    val analysisHistory by DataRepository.analysisHistory.collectAsState()'

text = text.replace(target, replacement)

with open(main_file, 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched correctly this time')
