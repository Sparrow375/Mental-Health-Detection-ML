"""
Trial Runner — Systematic parameter search over StudentLife dataset.

Tries multiple configurations of:
  - Baseline days: 7, 10, 14, 21, 28
  - Hybrid vs fixed baseline
  - Anomaly thresholds: 0.25 - 0.50
  - Evidence thresholds: 1.0 - 3.0
  - Prediction criteria
  - Bayesian kappa_0 values

Evaluates each trial against PHQ-9 ground truth and finds the best
configuration for sensitivity + specificity.

Ground truth strategy:
  - Primary: Use POST PHQ-9 (end-of-study) as the label for monitoring period
  - Secondary: Use MAX(pre, post) PHQ-9 for students with both scores
  - Tertiary: Use PRE PHQ-9 only for students without post scores
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from studentlife_loader import load_all_students, load_phq9, get_depression_labels, load_stress_ema
from simulation_engine import SimulationEngine, SimulationConfig, StudentResult
from metrics import compute_binary_metrics, compute_correlation_metrics, find_optimal_thresholds


# ============================================================================
# Trial Configurations to Test
# ============================================================================

def generate_trial_configs() -> List[SimulationConfig]:
    """Generate a highly focused clinical-priority grid of trial configurations."""
    configs = []
    trial_id = 0

    # ── Tier 1: Baseline Splits ──
    # Focused onboarding splits (5, 7, 10, 14 days)
    baseline_days_options = [5, 7, 10, 14]

    # Baseline Sweep (Hybrid Mode is superior, so we anchor on True)
    for bd in baseline_days_options:
        trial_id += 1
        configs.append(SimulationConfig(
            name=f"T{trial_id:03d}_bd{bd}_hybrid",
            baseline_days=bd,
            hybrid_mode=True,
        ))

    # ── Tier 2: Prediction Strategy Sweeps ──
    bd_sweeps = [5, 7, 10, 14]

    # 1. Mean Anomaly Strategy (Primary continuous scorer)
    for bd in bd_sweeps:
        for psc in [0.45, 0.50, 0.52, 0.55]:
            trial_id += 1
            configs.append(SimulationConfig(
                name=f"T{trial_id:03d}_strat_mean_anomaly_bd{bd}_psc{psc}_overridesTrue",
                baseline_days=bd,
                hybrid_mode=True,
                prediction_strategy="mean_anomaly",
                prediction_score_threshold=psc,
                clinical_overrides_enabled=True,
            ))

    # 2. Peak Evidence Strategy
    for bd in bd_sweeps:
        for pe in [40.0, 80.0]:
            trial_id += 1
            configs.append(SimulationConfig(
                name=f"T{trial_id:03d}_strat_peak_evidence_bd{bd}_pe{pe}_overridesTrue",
                baseline_days=bd,
                hybrid_mode=True,
                prediction_strategy="peak_evidence",
                prediction_evidence_threshold=pe,
                clinical_overrides_enabled=True,
            ))

    # 3. Sustained Days Strategy
    for bd in bd_sweeps:
        for ps in [20, 30]:
            trial_id += 1
            configs.append(SimulationConfig(
                name=f"T{trial_id:03d}_strat_sustained_days_bd{bd}_ps{ps}_overridesTrue",
                baseline_days=bd,
                hybrid_mode=True,
                prediction_strategy="sustained_days",
                prediction_sustained_threshold=ps,
                clinical_overrides_enabled=True,
            ))

    # 4. Idiographic Anomaly Strategy (Personalized SPC-like boundaries)
    for bd in bd_sweeps:
        for k in [1.3, 1.5, 1.8]:
            for ratio in [0.20, 0.25]:
                trial_id += 1
                configs.append(SimulationConfig(
                    name=f"T{trial_id:03d}_strat_idiographic_bd{bd}_k{k}_r{ratio}_overridesTrue",
                    baseline_days=bd,
                    hybrid_mode=True,
                    prediction_strategy="idiographic_anomaly",
                    prediction_score_threshold=k,
                    prediction_evidence_threshold=ratio,
                    clinical_overrides_enabled=True,
                ))

    # ── Tier 3: Bayesian Hyperparameters ──
    for kappa in [7.0, 14.0]:
        for bd in [5, 7]:
            trial_id += 1
            configs.append(SimulationConfig(
                name=f"T{trial_id:03d}_kappa{kappa}_bd{bd}",
                baseline_days=bd,
                hybrid_mode=True,
                kappa_0=kappa,
            ))

    # Strategy E: Extreme/Conservative/Aggressive Presets
    extreme_configs = [
        SimulationConfig(
            name=f"T{trial_id+1:03d}_aggressive",
            baseline_days=5,
            hybrid_mode=True,
            anomaly_threshold=0.25,
            prediction_evidence_threshold=15.0,
            prediction_sustained_threshold=10,
            prediction_score_threshold=0.42,
            clinical_overrides_enabled=True,
        ),
        SimulationConfig(
            name=f"T{trial_id+2:03d}_conservative",
            baseline_days=14,
            hybrid_mode=True,
            anomaly_threshold=0.50,
            prediction_evidence_threshold=90.0,
            prediction_sustained_threshold=45,
            prediction_score_threshold=0.55,
            clinical_overrides_enabled=True,
        ),
        SimulationConfig(
            name=f"T{trial_id+3:03d}_balanced",
            baseline_days=7,
            hybrid_mode=True,
            anomaly_threshold=0.35,
            prediction_evidence_threshold=40.0,
            prediction_sustained_threshold=25,
            prediction_score_threshold=0.50,
            evidence_decay=0.90,
            kappa_0=10.0,
            clinical_overrides_enabled=True,
        ),
    ]
    configs.extend(extreme_configs)

    return configs


# ============================================================================
# Ground Truth Assembly
# ============================================================================

def get_ground_truth_labels(
    results: List[StudentResult],
    strategy: str = "post",
    cutoff: int = 10,
) -> Tuple[List[bool], List[bool], List[float], List[str]]:
    """
    Extract ground truth and predictions from simulation results.

    Strategies:
        "post" - Use post-study PHQ-9 only
        "max" - Use max(pre, post) PHQ-9
        "pre" - Use pre-study PHQ-9 only
        "any" - Use post if available, else pre

    Returns:
        (y_true, y_pred, y_scores, uids) — filtered to students with ground truth.
    """
    y_true = []
    y_pred = []
    y_scores = []
    uids = []

    for r in results:
        if strategy == "post":
            if r.phq9_post is None:
                continue
            gt = r.phq9_post >= cutoff
        elif strategy == "max":
            scores = [s for s in [r.phq9_pre, r.phq9_post] if s is not None]
            if not scores:
                continue
            gt = max(scores) >= cutoff
        elif strategy == "pre":
            if r.phq9_pre is None:
                continue
            gt = r.phq9_pre >= cutoff
        elif strategy == "any":
            if r.phq9_post is not None:
                gt = r.phq9_post >= cutoff
            elif r.phq9_pre is not None:
                gt = r.phq9_pre >= cutoff
            else:
                continue
        else:
            continue

        y_true.append(gt)
        y_pred.append(r.predicted_depressed)
        y_scores.append(r.mean_anomaly_score)
        uids.append(r.uid)

    return y_true, y_pred, y_scores, uids


# ============================================================================
# Single Trial Evaluator
# ============================================================================

def evaluate_trial(
    config: SimulationConfig,
    all_data: Dict,
    phq9_scores: Dict,
    gt_strategy: str = "post",
) -> Dict:
    """
    Run one trial and evaluate against ground truth.

    Returns a dict with config, metrics, and detailed results.
    """
    engine = SimulationEngine(config)
    results = engine.simulate_all(all_data, phq9_scores)

    if not results:
        return {"config": config.name, "error": "No results", "n_students": 0}

    # Extract ground truth
    y_true, y_pred, y_scores, uids = get_ground_truth_labels(
        results, strategy=gt_strategy, cutoff=config.depression_cutoff
    )

    if len(y_true) < 5:
        return {"config": config.name, "error": "Too few labeled students", "n_students": len(y_true)}

    # Binary metrics
    binary = compute_binary_metrics(y_true, y_pred, y_scores)

    # Correlation metrics
    phq9_list = []
    anomaly_list = []
    evidence_list = []
    sustained_list = []
    ratio_list = []
    for r in results:
        score = None
        if gt_strategy == "post":
            score = r.phq9_post
        elif gt_strategy == "pre":
            score = r.phq9_pre
        elif gt_strategy == "any":
            score = r.phq9_post if r.phq9_post is not None else r.phq9_pre
        elif gt_strategy == "max":
            scores = [s for s in [r.phq9_pre, r.phq9_post] if s is not None]
            if scores:
                score = max(scores)

        if score is None:
            continue

        phq9_list.append(score)
        anomaly_list.append(r.mean_anomaly_score)
        evidence_list.append(r.peak_evidence)
        sustained_list.append(r.peak_sustained)
        ratio_list.append(r.anomaly_day_ratio)

    correlation = compute_correlation_metrics(phq9_list, anomaly_list, evidence_list)

    # Optimal thresholds
    optimal = find_optimal_thresholds(
        y_true, anomaly_list, evidence_list, sustained_list, ratio_list
    )

    # Compile trial result
    trial_result = {
        "config_name": config.name,
        "config": {
            "baseline_days": config.baseline_days,
            "hybrid_mode": config.hybrid_mode,
            "anomaly_threshold": config.anomaly_threshold,
            "evidence_decay": config.evidence_decay,
            "evidence_compounding": config.evidence_compounding,
            "prediction_evidence_threshold": config.prediction_evidence_threshold,
            "prediction_sustained_threshold": config.prediction_sustained_threshold,
            "prediction_score_threshold": config.prediction_score_threshold,
            "prediction_strategy": config.prediction_strategy,
            "kappa_0": config.kappa_0,
            "l1_exponent": config.l1_exponent,
            "l2_exponent": config.l2_exponent,
            "compactness_N": config.compactness_N,
            "compactness_threshold": config.compactness_threshold,
            "clinical_overrides_enabled": config.clinical_overrides_enabled,
        },
        "n_students_evaluated": len(results),
        "n_students_labeled": len(y_true),
        "gt_strategy": gt_strategy,
        "binary_metrics": binary,
        "correlation_metrics": correlation,
        "optimal_thresholds": optimal,
        "per_student": [
            {
                "uid": r.uid,
                "phq9_pre": r.phq9_pre,
                "phq9_post": r.phq9_post,
                "depressed_pre": r.depressed_pre,
                "depressed_post": r.depressed_post,
                "predicted_depressed": r.predicted_depressed,
                "prediction_confidence": round(r.prediction_confidence, 4),
                "mean_anomaly": round(r.mean_anomaly_score, 4),
                "peak_evidence": round(r.peak_evidence, 4),
                "peak_sustained": r.peak_sustained,
                "anomaly_day_ratio": round(r.anomaly_day_ratio, 4),
                "n_days": len(r.dates),
                "n_baseline": r.n_baseline_days,
                "n_monitoring": r.n_monitoring_days,
            }
            for r in results
        ],
    }

    # Composite ranking score (balanced sensitivity + specificity)
    sens = binary.get("sensitivity", 0)
    spec = binary.get("specificity", 0)
    f1 = binary.get("f1_score", 0)
    auc = binary.get("auc_roc", 0)
    if isinstance(auc, float) and not (auc != auc):  # not NaN
        trial_result["composite_score"] = round(0.3 * sens + 0.3 * spec + 0.2 * f1 + 0.2 * auc, 4)
    else:
        trial_result["composite_score"] = round(0.4 * sens + 0.4 * spec + 0.2 * f1, 4)

    return trial_result


# ============================================================================
# Full Trial Runner
# ============================================================================

def run_all_trials(
    output_dir: str = None,
    gt_strategies: List[str] = None,
) -> Dict:
    """
    Run all trial configurations and find the best one.

    Returns comprehensive results dict.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    if gt_strategies is None:
        gt_strategies = ["post", "any", "max"]

    print("=" * 70)
    print("MHealth StudentLife Validation — Trial Runner")
    print("=" * 70)

    # ── Step 1: Load data ──
    print("\n[1/4] Loading StudentLife dataset...")
    t0 = time.time()
    all_data = load_all_students()
    phq9_scores = load_phq9()
    stress_ema = load_stress_ema()

    print(f"  Loaded {len(all_data)} students with sensor data")
    print(f"  PHQ-9 scores for {len(phq9_scores)} students")
    print(f"  Stress EMA for {len(stress_ema)} students")
    print(f"  Data loading took {time.time() - t0:.1f}s")

    # ── Step 2: Generate trials ──
    configs = generate_trial_configs()
    print(f"\n[2/4] Running {len(configs)} trial configurations across {len(gt_strategies)} GT strategies...")

    # ── Step 3: Run trials ──
    all_results = {}
    best_overall = {"composite_score": -1}
    total = len(configs) * len(gt_strategies)
    done = 0

    for gt_strat in gt_strategies:
        strat_results = []
        best_for_strat = {"composite_score": -1}

        for config in configs:
            done += 1
            try:
                trial = evaluate_trial(config, all_data, phq9_scores, gt_strat)

                strat_results.append(trial)

                comp = trial.get("composite_score", 0)
                sens = trial.get("binary_metrics", {}).get("sensitivity", 0)
                spec = trial.get("binary_metrics", {}).get("specificity", 0)

                if comp > best_for_strat.get("composite_score", -1):
                    best_for_strat = trial

                if comp > best_overall.get("composite_score", -1):
                    best_overall = trial
                    best_overall["gt_strategy"] = gt_strat

                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] {config.name} ({gt_strat}): "
                          f"Sens={sens:.3f} Spec={spec:.3f} Comp={comp:.3f}")

            except Exception as e:
                strat_results.append({
                    "config_name": config.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

        all_results[gt_strat] = {
            "trials": strat_results,
            "best": best_for_strat,
        }

    # ── Step 4: Save results ──
    print(f"\n[3/4] Saving results to {output_dir}/...")

    # Save full results
    results_path = os.path.join(output_dir, "all_trial_results.json")
    # Filter out internal ROC/PR curve data for the JSON dump (too large)
    export_results = {}
    for strat, data in all_results.items():
        export_trials = []
        for t in data["trials"]:
            filtered = {k: v for k, v in t.items()
                        if not isinstance(v, dict) or not any(k2.startswith('_') for k2 in v)}
            # Also filter internal keys from binary_metrics
            if "binary_metrics" in filtered:
                filtered["binary_metrics"] = {
                    k: v for k, v in filtered["binary_metrics"].items()
                    if not k.startswith('_')
                }
            export_trials.append(filtered)
        export_results[strat] = {
            "trials": export_trials,
            "best": {k: v for k, v in data["best"].items()
                     if not (isinstance(v, dict) and any(k2.startswith('_') for k2 in v))}
        }
        if "binary_metrics" in export_results[strat]["best"]:
            export_results[strat]["best"]["binary_metrics"] = {
                k: v for k, v in export_results[strat]["best"]["binary_metrics"].items()
                if not k.startswith('_')
            }

    with open(results_path, 'w') as f:
        json.dump(export_results, f, indent=2, default=str)

    # Save best config
    best_path = os.path.join(output_dir, "best_config.json")
    best_export = {k: v for k, v in best_overall.items()
                   if not (isinstance(v, dict) and any(k2.startswith('_') for k2 in v if isinstance(k2, str)))}
    if "binary_metrics" in best_export:
        best_export["binary_metrics"] = {
            k: v for k, v in best_export["binary_metrics"].items()
            if not k.startswith('_')
        }
    with open(best_path, 'w') as f:
        json.dump(best_export, f, indent=2, default=str)

    # ── Print Summary ──
    print(f"\n[4/4] Trial Summary")
    print("=" * 70)

    for strat, data in all_results.items():
        best = data["best"]
        bm = best.get("binary_metrics", {})
        print(f"\n  GT Strategy: {strat}")
        print(f"    Best config: {best.get('config_name', '?')}")
        print(f"    Sensitivity: {bm.get('sensitivity', '?')}")
        print(f"    Specificity: {bm.get('specificity', '?')}")
        print(f"    F1 Score:    {bm.get('f1_score', '?')}")
        print(f"    AUC-ROC:     {bm.get('auc_roc', '?')}")
        print(f"    Composite:   {best.get('composite_score', '?')}")

    print(f"\n  Overall Best: {best_overall.get('config_name', '?')}")
    print(f"    GT Strategy: {best_overall.get('gt_strategy', '?')}")
    bm = best_overall.get("binary_metrics", {})
    print(f"    Sensitivity: {bm.get('sensitivity', '?')}")
    print(f"    Specificity: {bm.get('specificity', '?')}")
    print(f"    AUC-ROC:     {bm.get('auc_roc', '?')}")

    return {
        "all_results": all_results,
        "best_overall": best_overall,
        "all_data": all_data,
        "phq9_scores": phq9_scores,
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    results = run_all_trials(output_dir=output_dir)
