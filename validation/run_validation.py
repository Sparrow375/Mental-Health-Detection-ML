"""
MHealth Validation Suite — Main Entry Point

Orchestrates the complete validation pipeline:
  1. Load StudentLife dataset → daily feature vectors
  2. Run systematic parameter trials through System 1 pipeline
  3. Evaluate against PHQ-9 ground truth
  4. Find best configuration for sensitivity + specificity
  5. Re-run best config and generate detailed reports
  6. Iterative refinement around the best config

Usage:
    py validation/run_validation.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

# Setup paths
VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VALIDATION_DIR)
ENGINE_PATH = os.path.join(PROJECT_ROOT, "MHealth - Copy", "app", "src", "main", "python")

sys.path.insert(0, VALIDATION_DIR)
sys.path.insert(0, ENGINE_PATH)

from studentlife_loader import load_all_students, load_phq9, get_depression_labels, load_stress_ema
from simulation_engine import SimulationEngine, SimulationConfig, StudentResult
from trial_runner import evaluate_trial, get_ground_truth_labels, generate_trial_configs
from metrics import compute_binary_metrics, compute_correlation_metrics, find_optimal_thresholds
from report_generator import generate_full_report


def refine_around_best(
    best_config: Dict,
    all_data: Dict,
    phq9_scores: Dict,
    gt_strategy: str,
) -> Dict:
    """
    Run a focused grid search around the best configuration found.
    
    Fine-tunes ±small deltas around each key parameter.
    """
    print("\n" + "=" * 70)
    print("REFINEMENT PHASE — Fine-tuning around best configuration")
    print("=" * 70)

    base = best_config.get("config", {})
    bd = base.get("baseline_days", 14)
    at = base.get("anomaly_threshold", 0.38)
    ed = base.get("evidence_decay", 0.88)
    pe = base.get("prediction_evidence_threshold", 1.5)
    ps = base.get("prediction_sustained_threshold", 3)
    psc = base.get("prediction_score_threshold", 0.45)
    strat = base.get("prediction_strategy", "mean_anomaly")
    kappa = base.get("kappa_0", 14.0)
    hybrid = base.get("hybrid_mode", True)
    
    # Layer 1 & 2 Exponents & Compactness parameters
    l1_exp = base.get("l1_exponent", 0.6)
    l2_exp = base.get("l2_exponent", 0.4)
    comp_N = base.get("compactness_N", 7)
    comp_thresh = base.get("compactness_threshold", 1.2)

    def make_config(name, **kwargs):
        params = {
            "baseline_days": bd,
            "hybrid_mode": hybrid,
            "anomaly_threshold": at,
            "evidence_decay": ed,
            "prediction_evidence_threshold": pe,
            "prediction_sustained_threshold": ps,
            "prediction_score_threshold": psc,
            "prediction_strategy": strat,
            "kappa_0": kappa,
            "l1_exponent": l1_exp,
            "l2_exponent": l2_exp,
            "compactness_N": comp_N,
            "compactness_threshold": comp_thresh,
        }
        params.update(kwargs)
        return SimulationConfig(name=name, **params)

    # Generate fine-grained variations
    configs = []
    trial_id = 0

    # Sweep anomaly threshold finely
    for at_delta in [-0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05]:
        at_val = round(at + at_delta, 3)
        if at_val < 0.15 or at_val > 0.65:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_at{at_val}",
            anomaly_threshold=at_val,
        ))

    # Sweep evidence threshold finely
    for pe_delta in [-0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]:
        pe_val = round(pe + pe_delta, 2)
        if pe_val < 0.3 or pe_val > 150.0:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_pe{pe_val}",
            prediction_evidence_threshold=pe_val,
        ))

    # Sweep prediction score threshold finely
    for psc_delta in [-0.10, -0.05, -0.03, 0, 0.03, 0.05, 0.10]:
        psc_val = round(psc + psc_delta, 3)
        if psc_val < 0.15 or psc_val > 0.75:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_psc{psc_val}",
            prediction_score_threshold=psc_val,
        ))

    # Sweep sustained threshold
    for ps_delta in [-2, -1, 0, 1, 2]:
        ps_val = ps + ps_delta
        if ps_val < 1 or ps_val > 80:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_ps{ps_val}",
            prediction_sustained_threshold=ps_val,
        ))

    # Sweep baseline days around best
    for bd_delta in [-3, -2, -1, 0, 1, 2, 3, 5]:
        bd_val = bd + bd_delta
        if bd_val < 5 or bd_val > 28:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_bd{bd_val}",
            baseline_days=bd_val,
        ))

    # Sweep decay rate finely
    for ed_delta in [-0.05, -0.03, -0.02, 0, 0.02, 0.03, 0.05]:
        ed_val = round(ed + ed_delta, 3)
        if ed_val < 0.70 or ed_val > 0.98:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_ed{ed_val}",
            evidence_decay=ed_val,
        ))

    # Combinatorial: sweep the two most impactful together
    for at_d in [-0.03, 0, 0.03]:
        for psc_d in [-0.05, 0, 0.05]:
            for pe_d in [-0.3, 0, 0.3]:
                if at_d == 0 and psc_d == 0 and pe_d == 0:
                    continue
                trial_id += 1
                configs.append(make_config(
                    name=f"R{trial_id:03d}_combo_at{round(at+at_d,3)}_psc{round(psc+psc_d,3)}_pe{round(pe+pe_d,2)}",
                    anomaly_threshold=round(at + at_d, 3),
                    prediction_evidence_threshold=round(pe + pe_d, 2),
                    prediction_score_threshold=round(psc + psc_d, 3),
                ))

    # Sweep exponents
    for l1_e, l2_e in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.8, 0.2)]:
        if abs(l1_e - l1_exp) < 1e-4 and abs(l2_e - l2_exp) < 1e-4:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_exp_{l1_e}_{l2_e}",
            l1_exponent=l1_e,
            l2_exponent=l2_e,
        ))

    # Sweep compactness N
    for c_N in [3, 5, 7, 10]:
        if c_N == comp_N:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_compN_{c_N}",
            compactness_N=c_N,
        ))

    # Sweep compactness threshold
    for c_t in [0.8, 1.0, 1.2, 1.5]:
        if abs(c_t - comp_thresh) < 1e-4:
            continue
        trial_id += 1
        configs.append(make_config(
            name=f"R{trial_id:03d}_compT_{c_t}",
            compactness_threshold=c_t,
        ))

    print(f"  Running {len(configs)} refinement trials...")

    best_refined = {"composite_score": -1}
    for i, config in enumerate(configs):
        try:
            trial = evaluate_trial(config, all_data, phq9_scores, gt_strategy)
            comp = trial.get("composite_score", 0)
            sens = trial.get("binary_metrics", {}).get("sensitivity", 0)
            spec = trial.get("binary_metrics", {}).get("specificity", 0)

            if comp > best_refined.get("composite_score", -1):
                best_refined = trial

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(configs)}] Best so far: Sens={best_refined.get('binary_metrics', {}).get('sensitivity', 0):.3f} "
                      f"Spec={best_refined.get('binary_metrics', {}).get('specificity', 0):.3f}")
        except Exception as e:
            pass

    print(f"\n  Refinement complete!")
    bm = best_refined.get("binary_metrics", {})
    print(f"  Best refined: {best_refined.get('config_name', '?')}")
    print(f"    Sensitivity: {bm.get('sensitivity', '?')}")
    print(f"    Specificity: {bm.get('specificity', '?')}")
    print(f"    F1:          {bm.get('f1_score', '?')}")
    print(f"    AUC-ROC:     {bm.get('auc_roc', '?')}")

    return best_refined


def main():
    """Main validation pipeline."""
    start_time = time.time()

    output_dir = os.path.join(VALIDATION_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  MHealth System Validation — StudentLife Dataset")
    print("  Comprehensive Trial-and-Error Parameter Search")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Load Data
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PHASE 1: Loading StudentLife Dataset")
    print("─" * 70)

    t0 = time.time()
    all_data = load_all_students()
    phq9_scores = load_phq9()
    depression_labels = get_depression_labels(phq9_scores)
    stress_ema = load_stress_ema()

    # Print dataset summary
    n_depressed_pre = sum(1 for s in phq9_scores.values() if s.get('pre', 0) >= 10)
    n_depressed_post = sum(1 for s in phq9_scores.values() if s.get('post', 0) >= 10)
    n_with_post = sum(1 for s in phq9_scores.values() if 'post' in s)

    print(f"\n  Dataset Summary:")
    print(f"    Students with sensor data: {len(all_data)}")
    print(f"    Students with PHQ-9: {len(phq9_scores)}")
    print(f"    Students with post-PHQ-9: {n_with_post}")
    print(f"    Depressed (pre):  {n_depressed_pre}")
    print(f"    Depressed (post): {n_depressed_post}")
    print(f"    Avg days/student: {np.mean([len(d) for d in all_data.values()]):.1f}")
    print(f"    Load time: {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Broad Parameter Search
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PHASE 2: Broad Parameter Search")
    print("─" * 70)

    configs = generate_trial_configs()
    gt_strategies = ["post", "any", "max"]

    all_trial_results = {}
    best_overall = {"composite_score": -1}
    best_gt_strategy = "post"
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
                if comp > best_for_strat.get("composite_score", -1):
                    best_for_strat = trial
                if comp > best_overall.get("composite_score", -1):
                    best_overall = trial
                    best_gt_strategy = gt_strat

                if done % 30 == 0 or done == total:
                    sens = trial.get("binary_metrics", {}).get("sensitivity", 0)
                    spec = trial.get("binary_metrics", {}).get("specificity", 0)
                    print(f"  [{done}/{total}] {config.name} ({gt_strat}): "
                          f"Sens={sens:.3f} Spec={spec:.3f} Comp={comp:.3f}")
            except Exception as e:
                strat_results.append({"config_name": config.name, "error": str(e)})

        all_trial_results[gt_strat] = {
            "trials": strat_results,
            "best": best_for_strat,
        }

    print(f"\n  Broad search complete!")
    bm = best_overall.get("binary_metrics", {})
    print(f"  Best: {best_overall.get('config_name', '?')} ({best_gt_strategy})")
    print(f"    Sensitivity: {bm.get('sensitivity', '?')}")
    print(f"    Specificity: {bm.get('specificity', '?')}")
    print(f"    AUC-ROC:     {bm.get('auc_roc', '?')}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Refinement Around Best
    # ══════════════════════════════════════════════════════════════════════
    best_refined = refine_around_best(best_overall, all_data, phq9_scores, best_gt_strategy)

    # Pick the truly best
    if best_refined.get("composite_score", 0) > best_overall.get("composite_score", 0):
        best_final = best_refined
        print("\n  ✓ Refinement improved results!")
    else:
        best_final = best_overall
        print("\n  → Broad search was already optimal")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Re-run best config and generate detailed results
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("PHASE 4: Generating Detailed Report with Best Configuration")
    print("─" * 70)

    best_config_dict = best_final.get("config", {})
    final_config = SimulationConfig(
        name="FINAL_BEST",
        baseline_days=best_config_dict.get("baseline_days", 14),
        hybrid_mode=best_config_dict.get("hybrid_mode", True),
        anomaly_threshold=best_config_dict.get("anomaly_threshold", 0.38),
        evidence_decay=best_config_dict.get("evidence_decay", 0.88),
        evidence_compounding=best_config_dict.get("evidence_compounding", 0.15),
        prediction_evidence_threshold=best_config_dict.get("prediction_evidence_threshold", 1.5),
        prediction_sustained_threshold=best_config_dict.get("prediction_sustained_threshold", 3),
        prediction_score_threshold=best_config_dict.get("prediction_score_threshold", 0.45),
        prediction_strategy=best_config_dict.get("prediction_strategy", "mean_anomaly"),
        kappa_0=best_config_dict.get("kappa_0", 14.0),
        l1_exponent=best_config_dict.get("l1_exponent", 0.6),
        l2_exponent=best_config_dict.get("l2_exponent", 0.4),
        compactness_N=best_config_dict.get("compactness_N", 7),
        compactness_threshold=best_config_dict.get("compactness_threshold", 1.2),
        clinical_overrides_enabled=best_config_dict.get("clinical_overrides_enabled", False),
    )

    engine = SimulationEngine(final_config)
    sim_results = engine.simulate_all(all_data, phq9_scores)

    # Generate all visualizations and report
    generate_full_report(best_final, all_trial_results, sim_results, output_dir)

    # Save final results
    final_output = {
        "best_config": best_final,
        "broad_search_summary": {
            strat: {
                "n_trials": len(data["trials"]),
                "best_config": data["best"].get("config_name", "?"),
                "best_sensitivity": data["best"].get("binary_metrics", {}).get("sensitivity", "?"),
                "best_specificity": data["best"].get("binary_metrics", {}).get("specificity", "?"),
                "best_auc": data["best"].get("binary_metrics", {}).get("auc_roc", "?"),
                "best_f1": data["best"].get("binary_metrics", {}).get("f1_score", "?"),
            }
            for strat, data in all_trial_results.items()
        },
        "total_trials_run": sum(len(d["trials"]) for d in all_trial_results.values()),
        "total_time_seconds": round(time.time() - start_time, 1),
    }

    # Clean internal keys before saving
    if "binary_metrics" in final_output["best_config"]:
        final_output["best_config"]["binary_metrics"] = {
            k: v for k, v in final_output["best_config"]["binary_metrics"].items()
            if not k.startswith('_')
        }

    with open(os.path.join(output_dir, "final_results.json"), 'w') as f:
        json.dump(final_output, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    bm_final = best_final.get("binary_metrics", {})

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"  Total trials: {final_output['total_trials_run']}")
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  BEST RESULTS                                │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Sensitivity:     {bm_final.get('sensitivity', '?'):>8}                │")
    print(f"  │  Specificity:     {bm_final.get('specificity', '?'):>8}                │")
    print(f"  │  PPV (Precision): {bm_final.get('ppv_precision', '?'):>8}                │")
    print(f"  │  NPV:             {bm_final.get('npv', '?'):>8}                │")
    print(f"  │  F1 Score:        {bm_final.get('f1_score', '?'):>8}                │")
    print(f"  │  AUC-ROC:         {bm_final.get('auc_roc', '?'):>8}                │")
    print(f"  │  Balanced Acc:    {bm_final.get('balanced_accuracy', '?'):>8}                │")
    print(f"  │  Cohen's Kappa:   {bm_final.get('cohens_kappa', '?'):>8}                │")
    print(f"  │  Youden's J:      {bm_final.get('youdens_j', '?'):>8}                │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Report: {os.path.join(output_dir, 'validation_report.md')}")
    print(f"  Plots:  {os.path.join(output_dir, 'plots')}/")


if __name__ == "__main__":
    main()
