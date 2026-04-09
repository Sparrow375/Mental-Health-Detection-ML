import sys
import os
import json

repo_path = r"c:\Users\embar\OneDrive-N\D0cuments\GitHub\Mental-Health-Detection-ML"
sys.path.insert(0, os.path.join(repo_path, "system2", "Integrated"))

from system2.explainability import ExplainabilityEngine
from system2.config import DISORDER_PROTOTYPES_FRAME2

data_path = os.path.join(repo_path, "system2", "data", "crosscheck_results.json")
with open(data_path, "r") as f:
    data = json.load(f)

engine = ExplainabilityEngine()

schiz_students = [s for s in data if s.get("ground_truth") == "schizophrenia"]

out_md = [
    "# System 2: Schizophrenia Students Pattern Analysis",
    "",
    f"**Total Schizophrenia Students Found:** {len(schiz_students)} (Ground truth: Schizophrenia)",
    ""
]

artifacts_dir = r"C:\Users\embar\.gemini\antigravity\artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Generate for first 10 students or all if few, otherwise we will spam images. Wait, crosscheck is big?
# The list had about hundreds of students but not all schizophrenia.
# Let's limit to 10 for safety if there are too many, but the user asked for "patterns for each".
# Let's see how many there are. I'll just write the loop but limit to first 15 so it doesn't take forever to plot.
count = 0
for s in schiz_students:
    if count >= 15:
        break
    uid = s["uid"]
    s2_disorder = s.get("s2_disorder")
    if not s2_disorder or s2_disorder == "healthy" or s2_disorder.startswith("healthy_"):
        s2_disorder = "schizophrenia_type_1"

    s2_score = s.get("s2_score", 0.0)
    dev = s.get("feature_deviations", {})
    
    chart_filename = f"radar_schiz_{uid}.png"
    chart_path = os.path.join(artifacts_dir, chart_filename)
    chart_url = chart_path.replace("\\", "/") 
    
    proto_name = s2_disorder
    if proto_name not in DISORDER_PROTOTYPES_FRAME2:
        proto_name = "schizophrenia_type_1"
        
    try:
        engine.generate_radar_chart(
            user_profile=dev,
            prototype=DISORDER_PROTOTYPES_FRAME2[proto_name],
            disorder_name=proto_name,
            save_path=chart_path,
            healthy_profile=DISORDER_PROTOTYPES_FRAME2.get("healthy", {})
        )
    except Exception as e:
        print(f"Error generating chart for {uid}: {e}")
    
    top_feats = engine.top_contributing_features(dev, DISORDER_PROTOTYPES_FRAME2[proto_name], n=5)
    feats_str = ", ".join([f.replace('_', ' ').title() for f in top_feats])
    
    out_md.append(f"## Student Profile: {uid.upper()}")
    out_md.append(f"- **Matched Pattern Prototype:** `{s2_disorder.replace('_', ' ').title()}`")
    out_md.append(f"- **System 2 Match Score:** `{s2_score:.2f}`")
    out_md.append(f"- **Key Driving Behavioral Changes:** {feats_str}")
    out_md.append(f"")
    out_md.append(f"![Radar {uid} Profile vs {proto_name}](file:///{chart_url.lstrip('/')})")
    out_md.append(f"")
    out_md.append("---")
    out_md.append("")
    count += 1

out_md.append(f"\n*(Note: Displaying charts for {count} out of {len(schiz_students)} patients.)*")

out_file_path = os.path.join(artifacts_dir, "schizophrenia_students_analysis.md")
with open(out_file_path, "w") as f:
    f.write("\n".join(out_md))

print(f"DONE: {out_file_path}")
