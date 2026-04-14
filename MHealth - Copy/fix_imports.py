import re

main_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\MainActivity.kt'
with open(main_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Add to HomeScreen
if 'val s1ProfileJson by DataRepository' not in text:
    homescreen_target = 'val vector by DataRepository.latestVector.collectAsState()'
    homescreen_replacement = '''val vector by DataRepository.latestVector.collectAsState()
    val s1ProfileJson by DataRepository.s1ProfileJson.collectAsState()
    val analysisResult by DataRepository.latestAnalysisResult.collectAsState()
    val analysisHistory by DataRepository.analysisHistory.collectAsState()'''
    
    # only replace the first occurrence in HomeScreen
    hs_idx = text.find('fun HomeScreen()')
    if hs_idx != -1:
        v_idx = text.find(homescreen_target, hs_idx)
        if v_idx != -1:
            text = text[:v_idx] + homescreen_replacement + text[v_idx + len(homescreen_target):]

# Also ensure java class Operator compareTo is fixed?
# MainActivity.kt:1266:49 'operator' modifier is required on 'FirNamedFunctionSymbol kotlin/compareTo'
# Wait! In SparklineLabel:
# `if (evidenceHistory.count() >= 2)`
# The error was: `MainAcitivty.kt:1266:49 'operator' modifier is required`
# Because evidenceHistory.count() is a function now. Let's just change evidenceHistory.count() to evidenceHistory.size
text = text.replace('evidenceHistory.count()', 'evidenceHistory.size')

with open(main_file, 'w', encoding='utf-8') as f:
    f.write(text)

print('Patched MainActivity.kt')

mon_file = r'c:\Users\SRIRAM\Documents\GitHub\Mental-Health-Detection-ML\MHealth - Copy\app\src\main\java\com\example\mhealth\ui\screens\MonitorScreen.kt'
with open(mon_file, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('analysisResult != null', 'latestResult != null')
text = text.replace('analysisResult?.alertLevel', 'latestResult?.alertLevel')
text = text.replace('alertColorForLevel(it.alertLevel)', 'alertColor(it.alertLevel)')

with open(mon_file, 'w', encoding='utf-8') as f:
    f.write(text)
    
print('Patched MonitorScreen.kt')
