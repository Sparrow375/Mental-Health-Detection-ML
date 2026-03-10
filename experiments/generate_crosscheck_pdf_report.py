"""
Generate PDF Report for CrossCheck Simulation Results
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import os

class CrossCheckReportGenerator:
    def __init__(self, results_file='experiments/crosscheck_results.json'):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.results = self.data['results']
        self.df = pd.DataFrame(self.results)
        
    def generate_report(self, output_file='experiments/CrossCheck_Analysis_Report.pdf'):
        print(f"Generating PDF report: {output_file}...")
        
        with PdfPages(output_file) as pdf:
            self._plot_title_page(pdf)
            self._plot_summary_stats(pdf)
            self._plot_correlation_analysis(pdf)
            self._plot_anomaly_distribution(pdf)
            self._plot_individual_summaries(pdf)
            
        print(f"✓ Saved PDF Report: {output_file}")
        return output_file

    def _plot_title_page(self, pdf):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        ax.text(0.5, 0.85, 'System 1 Mental Health Detection', 
                ha='center', va='top', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.80, 'Validation on CrossCheck Dataset',
                ha='center', va='top', fontsize=18)
        
        ax.text(0.5, 0.72, f'Analysis Date: {self.data["analysis_date"]}',
                ha='center', va='top', fontsize=12, style='italic')
        
        summary_text = f"""
EXECUTIVE SUMMARY

Total Users Analyzed: {len(self.results)}
Analysis Period: ~6 months longitudinal study

Key Findings:
• Anomalies Detected: {sum(self.df['anomaly_detected'])} users
• Average Anomaly Score: {self.df['anomaly_score'].mean():.3f}
• Detection Rate: {sum(self.df['anomaly_detected'])/len(self.df)*100:.1f}%

This report evaluates how System 1 (Anomaly Detection) correlates with 
EMA-reported depression symptoms in the CrossCheck clinical population.
"""
        ax.text(0.5, 0.55, summary_text,
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='azure', alpha=0.3),
                family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _plot_summary_stats(self, pdf):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Detection Status
        detected = self.df['anomaly_detected'].value_counts()
        axes[0, 0].pie(detected, labels=detected.index.map({True: 'Anomaly', False: 'Normal'}), 
                      autopct='%1.1f%%', colors=['coral', 'lightgreen'])
        axes[0, 0].set_title('Status Distribution')
        
        # Patterns
        patterns = self.df['pattern'].value_counts()
        sns.barplot(x=patterns.index, y=patterns.values, ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Detected Patterns')
        axes[0, 1].set_ylabel('Count')
        
        # Anomaly Score vs EMA Depressed
        valid_ema = self.df.dropna(subset=['ema_depressed_avg'])
        if len(valid_ema) > 1:
            sns.regplot(x='ema_depressed_avg', y='anomaly_score', data=valid_ema, ax=axes[1, 0])
            corr = np.corrcoef(valid_ema['ema_depressed_avg'], valid_ema['anomaly_score'])[0, 1]
            axes[1, 0].set_title(f'EMA Depressed vs Anomaly Score (r={corr:.2f})')
        else:
            axes[1, 0].text(0.5, 0.5, "Insufficient EMA data for correlation", ha='center')
            
        # Monitoring Days
        sns.histplot(self.df['monitoring_days'], ax=axes[1, 1], kde=True)
        axes[1, 1].set_title('Duration of Monitoring (Days)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle('CrossCheck Dataset Statistics', fontsize=16, fontweight='bold')
        pdf.savefig(fig)
        plt.close()

    def _plot_correlation_analysis(self, pdf):
        # Placeholder for more detailed correlation if needed
        pass

    def _plot_anomaly_distribution(self, pdf):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='anomaly_detected', y='anomaly_score', data=self.df, ax=ax, palette='Set2')
        ax.set_title('Anomaly Score Distribution by Detection Status')
        ax.set_xticklabels(['Normal', 'Anomaly Detected'])
        pdf.savefig(fig)
        plt.close()

    def _plot_individual_summaries(self, pdf):
        # Individual results table (paginated)
        items_per_page = 20
        for i in range(0, len(self.results), items_per_page):
            chunk = self.results[i:i+items_per_page]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            table_data = [['User ID', 'Avg EMA Depr', 'Anomaly Score', 'Detected?', 'Pattern', 'Days']]
            for r in chunk:
                ema = f"{r['ema_depressed_avg']:.2f}" if r['ema_depressed_avg'] is not None else 'N/A'
                table_data.append([
                    r['user_id'],
                    ema,
                    f"{r['anomaly_score']:.3f}",
                    'YES' if r['anomaly_detected'] else 'NO',
                    r['pattern'],
                    r['monitoring_days']
                ])
                
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            
            # Header color
            for j in range(6):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', weight='bold')
            
            ax.set_title(f'Individual Results (Page {i//items_per_page + 1})', fontsize=14, fontweight='bold')
            pdf.savefig(fig)
            plt.close()

if __name__ == "__main__":
    gen = CrossCheckReportGenerator()
    gen.generate_report()
