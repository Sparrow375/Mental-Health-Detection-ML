"""
Master Pipeline Script for CrossCheck Experiment
"""

import os
import sys

def main():
    print("="*80)
    print("CROSSCHECK DATASET ANALYSIS PIPELINE")
    print("="*80)
    
    # 1. Run Simulation
    print("\nSTEP 1: Running System 1 Simulation on CrossCheck data...")
    from run_crosscheck_simulation import run_simulation
    data_path = r"F:\Avaneesh\download\archive\CrossCheck_Daily_Data.csv"
    results_file = run_simulation(data_path, output_dir='experiments')
    
    # 2. Generate PDF Report
    print("\nSTEP 2: Generating PDF Report...")
    from generate_crosscheck_pdf_report import CrossCheckReportGenerator
    generator = CrossCheckReportGenerator(results_file)
    pdf_path = generator.generate_report('experiments/CrossCheck_Analysis_Report.pdf')
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print(f"Results JSON: {results_file}")
    print(f"PDF Report:    {pdf_path}")
    print("="*80)

if __name__ == "__main__":
    main()
