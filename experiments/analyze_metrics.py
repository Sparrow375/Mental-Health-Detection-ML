import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def analyze():
    with open('experiments/crosscheck_results.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    df = pd.DataFrame(results)
    df_valid = df.dropna(subset=['ema_depressed_avg']).copy()
    
    if len(df_valid) < 2:
        return

    corr, p_value = pearsonr(df_valid['anomaly_score'], df_valid['ema_depressed_avg'])
    
    ground_truth_threshold = 0.5
    df_valid['is_depressed_gt'] = df_valid['ema_depressed_avg'] > ground_truth_threshold
    
    tp = int(len(df_valid[(df_valid['anomaly_detected'] == True) & (df_valid['is_depressed_gt'] == True)]))
    fp = int(len(df_valid[(df_valid['anomaly_detected'] == True) & (df_valid['is_depressed_gt'] == False)]))
    tn = int(len(df_valid[(df_valid['anomaly_detected'] == False) & (df_valid['is_depressed_gt'] == False)]))
    fn = int(len(df_valid[(df_valid['anomaly_detected'] == False) & (df_valid['is_depressed_gt'] == True)]))
    
    precision = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recall = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
    f1 = float(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
    accuracy = float((tp + tn) / len(df_valid))
    
    metrics = {
        "pearson_correlation": corr,
        "p_value": p_value,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    with open('experiments/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    analyze()
