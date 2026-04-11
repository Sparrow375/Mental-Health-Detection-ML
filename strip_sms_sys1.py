import os

path = r"MHealth - Copy\app\src\main\python\system1.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "texts_per_day" in line or "response_time_minutes" in line:
        continue
    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
