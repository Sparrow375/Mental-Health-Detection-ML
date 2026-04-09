import sys
import os
import json

repo_path = r"c:\Users\embar\OneDrive-N\D0cuments\GitHub\Mental-Health-Detection-ML"
sys.path.insert(0, os.path.join(repo_path, "system2", "Integrated"))

from system2.explainability import ExplainabilityEngine
from system2.config import DISORDER_PROTOTYPES_FRAME2

artifacts_dir = r"C:\Users\embar\.gemini\antigravity\artifacts"
os.makedirs(artifacts_dir, exist_ok=True)
engine = ExplainabilityEngine()

def process_dataset(filepath, disorder_label, s2_prefix, out_filename, title):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Filter for True Positives
    valid_students = []
    for s in data:
        gt = s.get("ground_truth")
        s2 = s.get("s2_disorder", "")
        if gt == disorder_label and s2 and s2.startswith(s2_prefix):
            valid_students.append(s)
            
    out_md = [
        f"# {title}",
        "",
        f"**Total True Positive {disorder_label.title()} Students Found:** {len(valid_students)} (Ground truth: {disorder_label.title()}, System 2 matched {s2_prefix.title()})",
        ""
    ]
    
    count = 0
    for s in valid_students:
        if count >= 15:
            break
        uid = s["uid"]
        s2_disorder = s["s2_disorder"]
        s2_score = s.get("s2_score", 0.0)
        dev = s.get("feature_deviations", {})
        
        chart_filename = f"radar_true_{s2_prefix}_{uid}.png"
        chart_path = os.path.join(artifacts_dir, chart_filename)
        chart_url = chart_path.replace("\\", "/") 
        
        try:
            engine.generate_radar_chart(
                user_profile=dev,
                prototype=DISORDER_PROTOTYPES_FRAME2[s2_disorder],
                disorder_name=s2_disorder,
                save_path=chart_path,
                healthy_profile=DISORDER_PROTOTYPES_FRAME2.get("healthy", {})
            )
        except Exception as e:
            print(f"Error generating chart for {uid}: {e}")
            continue
        
        top_feats = engine.top_contributing_features(dev, DISORDER_PROTOTYPES_FRAME2[s2_disorder], n=5)
        feats_str = ", ".join([f.replace('_', ' ').title() for f in top_feats])
        
        out_md.append(f"## Student Profile: {uid.upper()}")
        out_md.append(f"- **System 2 Matched Prototype:** `{s2_disorder.replace('_', ' ').title()}`")
        out_md.append(f"- **System 2 Match Score:** `{s2_score:.2f}`")
        out_md.append(f"- **Key Driving Behavioral Changes:** {feats_str}")
        out_md.append("")
        out_md.append(f"![Radar {uid} Profile vs {s2_disorder}](file:///{chart_url.lstrip('/')})")
        out_md.append("")
        out_md.append("---")
        out_md.append("")
        count += 1

    if len(valid_students) > 15:
        out_md.append(f"\n*(Note: Displaying charts for {count} out of {len(valid_students)} true positive patients.)*")

    out_file_path = os.path.join(artifacts_dir, out_filename)
    with open(out_file_path, "w") as f:
        f.write("\n".join(out_md))

    print(f"DONE: {out_file_path}")

dep_path = os.path.join(repo_path, "system2", "data", "studentlife_results.json")
schiz_path = os.path.join(repo_path, "system2", "data", "crosscheck_results.json")

process_dataset(dep_path, "depressed", "depression", "true_dep_analysis.md", "System 2: Verified Depression Pattern Analysis")
process_dataset(schiz_path, "schizophrenia", "schizophrenia", "true_schiz_analysis.md", "System 2: Verified Schizophrenia Pattern Analysis")
