import sys
import os
import json

repo_path = r"c:\Users\embar\OneDrive-N\D0cuments\GitHub\Mental-Health-Detection-ML"
sys.path.insert(0, repo_path)

from system2.explainability import ExplainabilityEngine
from system2.config import DISORDER_PROTOTYPES_FRAME2

data_path = os.path.join(repo_path, "system2", "data", "studentlife_results.json")
with open(data_path, "r") as f:
    data = json.load(f)

engine = ExplainabilityEngine()

depressed_students = [s for s in data if s.get("ground_truth") == "depressed"]

out_md = [
    "# System 2: Depressed Students Pattern Analysis",
    "",
    f"**Total Depressed Students Found:** {len(depressed_students)} (Ground truth: Depressed)",
    ""
]

artifacts_dir = os.path.join(repo_path, "system2")
os.makedirs(artifacts_dir, exist_ok=True)

for s in depressed_students:
    uid = s["uid"]
    s2_disorder = s.get("s2_disorder")
    if not s2_disorder or s2_disorder == "healthy" or s2_disorder.startswith("healthy_"):
        s2_disorder = "depression_type_1"

    s2_score = s.get("s2_score", 0.0)
    dev = s.get("feature_deviations", {})
    
    chart_filename = f"radar_{uid}_v6.png"
    chart_path = os.path.join(artifacts_dir, chart_filename)
    chart_url = chart_path.replace("\\", "/") 
    
    proto_name = s2_disorder
    if proto_name not in DISORDER_PROTOTYPES_FRAME2:
        proto_name = "depression_type_1"
        
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
    
    top_feats = engine.top_contributing_features(dev, DISORDER_PROTOTYPES_FRAME2[proto_name], n=10)  # Request more to filter zeros out
    
    out_md.append(f"## Student Profile: {uid.upper()}")
    out_md.append(f"### System 1: Anomaly Triggered")
    out_md.append(f"> **Baseline Break Detected**: System 1 flagged a structural deviation.")
    out_md.append(f"- **Global Anomaly Score:** `{s.get('s1_final_score', 'N/A')} SD`")
    out_md.append(f"- **Sustained Deviation:** `{s.get('s1_max_sustained_days', 'N/A')} days`")
    out_md.append(f"- **Trigger Pattern:** `{str(s.get('s1_pattern', 'N/A')).title()}`")
    out_md.append(f"")
    out_md.append(f"### System 2: Prototype Classification")
    out_md.append(f"- **Matched Pattern Prototype:** `{s2_disorder.replace('_', ' ').title()}`")
    out_md.append(f"- **System 2 Match Score:** `{s2_score:.2f}` / 1.0")
    out_md.append(f"")
    
    out_md.append(f"#### Top Behavioral Changes Analysis")
    out_md.append(f"| Feature | Anomaly Deviation | Pattern Match Rate |")
    out_md.append(f"|---|---|---|")
    
    healthy_p = DISORDER_PROTOTYPES_FRAME2.get("healthy", {})
    proto_p = DISORDER_PROTOTYPES_FRAME2[proto_name]
    for feat in top_feats:
        deviation = dev.get(feat, 0.0)
        if abs(deviation) < 0.01:
            continue  # Skip features with exactly 0 deviation
            
        expected = proto_p.get(feat, 0.0)
        
        f_name = feat.replace('_', ' ').title()
        
        if deviation * expected > 0:
            match_rate = f"Matches Expected ({expected:+.1f} SD)"
        elif expected == 0:
            match_rate = "Not core to pattern"
        else:
            match_rate = f"Diverges (Expected {expected:+.1f} SD)"
            
        out_md.append(f"| {f_name} | **{deviation:+.2f} SD** | {match_rate} |")

    out_md.append(f"")
    out_md.append(f"![Radar {uid} Profile vs {proto_name}]({chart_filename})")
    out_md.append(f"")
    out_md.append("---")
    out_md.append("")

out_file_path = os.path.join(artifacts_dir, "depressed_students_analysis.md")
with open(out_file_path, "w") as f:
    f.write("\n".join(out_md))

print(f"DONE: {out_file_path}")
