import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_method_comparison(df, output_dir='results/figures'):
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Recall comparison
    recall_data = df[df['k'] == 10][['method', 'avg_recall@10']].set_index('method')
    recall_data.plot(kind='bar', ax=ax1, legend=False, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax1.set_title('Recall@10 by Method')
    ax1.set_ylabel('Recall@10')
    ax1.set_xlabel('')
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)

    # Latency comparison
    latency_data = df[df['k'] == 10][['method', 'p50_latency']].set_index('method')
    latency_data.plot(kind='bar', ax=ax2, legend=False, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax2.set_title('P50 Latency by Method')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_xlabel('')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/method_comparison.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/method_comparison.png')
    plt.close()


def plot_selectivity_analysis(df, output_dir='results/figures'):
    os.makedirs(output_dir, exist_ok=True)

    df_k10 = df[df['k'] == 10].copy()
    bins = df_k10['selectivity_bin'].unique()
    bin_order = ['0.01-0.05', '0.05-0.15', '0.15-0.3', '0.3-1.0']
    bins = [b for b in bin_order if b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Recall vs Selectivity
    for method in ['Oracle', 'PreFilter', 'PostFilter', 'HybridSearch']:
        method_data = df_k10[df_k10['method'] == method]
        recalls = [method_data[method_data['selectivity_bin'] == b]['avg_recall@10'].values[0]
                   if len(method_data[method_data['selectivity_bin'] == b]) > 0 else 0
                   for b in bins]
        ax1.plot(range(len(bins)), recalls, marker='o', label=method, linewidth=2)

    ax1.set_xticks(range(len(bins)))
    ax1.set_xticklabels(bins, rotation=45)
    ax1.set_xlabel('Selectivity Range')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Recall@10 vs Selectivity')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.1])

    # Latency vs Selectivity
    for method in ['Oracle', 'PreFilter', 'PostFilter', 'HybridSearch']:
        method_data = df_k10[df_k10['method'] == method]
        latencies = [method_data[method_data['selectivity_bin'] == b]['p50_latency'].values[0]
                     if len(method_data[method_data['selectivity_bin'] == b]) > 0 else 0
                     for b in bins]
        ax2.plot(range(len(bins)), latencies, marker='o', label=method, linewidth=2)

    ax2.set_xticks(range(len(bins)))
    ax2.set_xticklabels(bins, rotation=45)
    ax2.set_xlabel('Selectivity Range')
    ax2.set_ylabel('P50 Latency (ms)')
    ax2.set_title('Latency vs Selectivity')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/selectivity_analysis.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/selectivity_analysis.png')
    plt.close()


def print_summary(comparison_df, selectivity_df):
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nMethod Comparison (k=10):")
    summary = comparison_df[comparison_df['k'] == 10][
        ['method', 'avg_recall@10', 'p50_latency', 'mean_latency', 'n_queries']
    ]
    print(summary.to_string(index=False))

    print("\n\nSelectivity Analysis (k=10):")
    sel_summary = selectivity_df[selectivity_df['k'] == 10].pivot_table(
        index='selectivity_bin',
        columns='method',
        values='avg_recall@10'
    )
    print(sel_summary)


def main():
    results_dir = 'results'

    print("="*60)
    print("Results Analysis")
    print("="*60)

    # Load results
    comparison_df = pd.read_csv(f'{results_dir}/comparison.csv')
    selectivity_df = pd.read_csv(f'{results_dir}/selectivity_analysis.csv')

    # Print summary
    print_summary(comparison_df, selectivity_df)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_method_comparison(comparison_df)
    plot_selectivity_analysis(selectivity_df)

    print("\n" + "="*60)
    print("Analysis complete!")


if __name__ == '__main__':
    main()
