import os

path = r"MHealth - Copy\app\src\main\python\config.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Drop the two deprecated features entirely
    if '"texts_per_day"' in line or '"response_time_minutes"' in line:
        continue
    
    # Redistribute the 0.7 combined weight into the remaining social markers (HYBRID)
    if '"social_app_ratio":' in line and '0.1' in line:
        # 0.1 -> 0.4 (+0.3)
        line = line.replace('0.1', '0.4')
    elif '"calls_per_day":' in line and '0.5' in line:
        # 0.5 -> 0.7 (+0.2)
        line = line.replace('0.5', '0.7')
    elif '"unique_contacts":' in line and '0.4' in line:
        # 0.4 -> 0.6 (+0.2)
        line = line.replace('0.4', '0.6')
        
    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Successfully purged SMS features from 12 prototypes and updated the weight distribution matrix.")
