import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_exp1_recall(comparison_df, output_dir='results/figures'):
    """Experiment 1: Recall@k by Method"""
    os.makedirs(output_dir, exist_ok=True)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    colors = ['#3498db', '#e74c3c', '#f39c12']
    k_values = [10, 20, 50]

    # Prepare data
    data = {}
    for k in k_values:
        data[k] = []
        for method in methods:
            row = comparison_df[(comparison_df['method'] == method) & (comparison_df['k'] == k)]
            if len(row) > 0:
                data[k].append(row[f'avg_recall@{k}'].values[0])
            else:
                data[k].append(0)

    # Plot
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(k_values):
        ax.bar(x + i*width, data[k], width, label=f'k={k}', color=colors[i], alpha=0.8)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Recall@k', fontsize=11)
    ax.set_title('Recall@k by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_recall_by_method.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp1_recall_by_method.png')
    plt.close()


def plot_exp1_p50_latency(comparison_df, output_dir='results/figures'):
    """Experiment 1: P50 Latency by Method"""
    os.makedirs(output_dir, exist_ok=True)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    colors = ['#3498db', '#e74c3c', '#f39c12']
    k_values = [10, 20, 50]

    # Prepare data
    data = {}
    for k in k_values:
        data[k] = []
        for method in methods:
            row = comparison_df[(comparison_df['method'] == method) & (comparison_df['k'] == k)]
            if len(row) > 0:
                data[k].append(row['p50_latency'].values[0])
            else:
                data[k].append(0)

    # Plot
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(k_values):
        ax.bar(x + i*width, data[k], width, label=f'k={k}', color=colors[i], alpha=0.8)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('P50 Latency (ms)', fontsize=11)
    ax.set_title('P50 Latency by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_p50_latency_by_method.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp1_p50_latency_by_method.png')
    plt.close()


def plot_exp1_p95_latency(comparison_df, output_dir='results/figures'):
    """Experiment 1: P95 Latency by Method"""
    os.makedirs(output_dir, exist_ok=True)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    colors = ['#3498db', '#e74c3c', '#f39c12']
    k_values = [10, 20, 50]

    # Prepare data
    data = {}
    for k in k_values:
        data[k] = []
        for method in methods:
            row = comparison_df[(comparison_df['method'] == method) & (comparison_df['k'] == k)]
            if len(row) > 0:
                data[k].append(row['p95_latency'].values[0])
            else:
                data[k].append(0)

    # Plot
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(k_values):
        ax.bar(x + i*width, data[k], width, label=f'k={k}', color=colors[i], alpha=0.8)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('P95 Latency (ms)', fontsize=11)
    ax.set_title('P95 Latency by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_p95_latency_by_method.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp1_p95_latency_by_method.png')
    plt.close()


def plot_exp1_mean_latency(comparison_df, output_dir='results/figures'):
    """Experiment 1: Mean Latency by Method"""
    os.makedirs(output_dir, exist_ok=True)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    colors = ['#3498db', '#e74c3c', '#f39c12']
    k_values = [10, 20, 50]

    # Prepare data
    data = {}
    for k in k_values:
        data[k] = []
        for method in methods:
            row = comparison_df[(comparison_df['method'] == method) & (comparison_df['k'] == k)]
            if len(row) > 0:
                data[k].append(row['mean_latency'].values[0])
            else:
                data[k].append(0)

    # Plot
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(k_values):
        ax.bar(x + i*width, data[k], width, label=f'k={k}', color=colors[i], alpha=0.8)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Mean Latency (ms)', fontsize=11)
    ax.set_title('Mean Latency by Method', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_mean_latency_by_method.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp1_mean_latency_by_method.png')
    plt.close()


def plot_exp2_recall_vs_selectivity(selectivity_df, output_dir='results/figures'):
    """Experiment 2: Recall@20 vs Selectivity"""
    os.makedirs(output_dir, exist_ok=True)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    colors_map = {'PreFilter': '#3498db', 'PostFilter': '#e74c3c', 'HybridSearch': '#f39c12'}

    bin_order = ['0.01-0.05', '0.05-0.15', '0.15-0.3', '0.3-1.0']
    bin_labels = ['Very Selective\n(1-5%)', 'Selective\n(5-15%)',
                  'Medium\n(15-30%)', 'High\n(>30%)']

    # Use only k=20
    df_k20 = selectivity_df[selectivity_df['k'] == 20]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        data = df_k20[df_k20['method'] == method]
        recalls = []
        for b in bin_order:
            matching = data[data['selectivity_bin'] == b]
            if len(matching) > 0:
                recalls.append(matching['avg_recall@20'].values[0])
            else:
                recalls.append(np.nan)

        ax.plot(range(len(bin_order)), recalls, marker='o',
                label=method, linewidth=2.5, markersize=8,
                color=colors_map[method])

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('Recall@20', fontsize=11)
    ax.set_title('Recall@20 vs Selectivity', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_recall_vs_selectivity.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp2_recall_vs_selectivity.png')
    plt.close()


def plot_exp2_latency_vs_selectivity(selectivity_df, output_dir='results/figures'):
    """Experiment 2: P50 Latency vs Selectivity"""
    os.makedirs(output_dir, exist_ok=True)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    colors_map = {'PreFilter': '#3498db', 'PostFilter': '#e74c3c', 'HybridSearch': '#f39c12'}

    bin_order = ['0.01-0.05', '0.05-0.15', '0.15-0.3', '0.3-1.0']
    bin_labels = ['Very Selective\n(1-5%)', 'Selective\n(5-15%)',
                  'Medium\n(15-30%)', 'High\n(>30%)']

    # Use only k=20
    df_k20 = selectivity_df[selectivity_df['k'] == 20]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        data = df_k20[df_k20['method'] == method]
        latencies = []
        for b in bin_order:
            matching = data[data['selectivity_bin'] == b]
            if len(matching) > 0:
                latencies.append(matching['p50_latency'].values[0])
            else:
                latencies.append(np.nan)

        ax.plot(range(len(bin_order)), latencies, marker='o',
                label=method, linewidth=2.5, markersize=8,
                color=colors_map[method])

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('P50 Latency (ms)', fontsize=11)
    ax.set_title('P50 Latency vs Selectivity (k=20)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_latency_vs_selectivity.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp2_latency_vs_selectivity.png')
    plt.close()


def plot_exp2_query_distribution(selectivity_df, output_dir='results/figures'):
    """Experiment 2: Query Distribution by Selectivity"""
    os.makedirs(output_dir, exist_ok=True)

    bin_order = ['0.01-0.05', '0.05-0.15', '0.15-0.3', '0.3-1.0']
    bin_labels = ['Very Selective\n(1-5%)', 'Selective\n(5-15%)',
                  'Medium\n(15-30%)', 'High\n(>30%)']

    # Use k=20 to get query counts
    df_k20 = selectivity_df[selectivity_df['k'] == 20]
    bin_counts = df_k20.groupby('selectivity_bin')['n_queries'].first()
    counts = [bin_counts[b] if b in bin_counts.index else 0 for b in bin_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(bin_order)), counts, color='#95a5a6', alpha=0.7)

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('Number of Queries', fontsize=11)
    ax.set_title('Query Distribution by Selectivity', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, count in enumerate(counts):
        ax.text(i, count + max(counts)*0.02, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_query_distribution.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp2_query_distribution.png')
    plt.close()


def print_summary(comparison_df, selectivity_df):
    """Print text summary of results"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: METHOD COMPARISON")
    print("="*70)

    methods = ['PreFilter', 'PostFilter', 'HybridSearch']
    for k in [10, 20, 50]:
        print(f"\nk = {k}:")
        df = comparison_df[comparison_df['k'] == k]
        df = df[df['method'].isin(methods)]
        cols = ['method', f'avg_recall@{k}', 'p50_latency', 'p95_latency', 'mean_latency']
        summary = df[cols].copy()
        summary.columns = ['Method', f'Recall@{k}', 'P50 (ms)', 'P95 (ms)', 'Mean (ms)']
        print(summary.to_string(index=False))

    print("\n" + "="*70)
    print("EXPERIMENT 2: SELECTIVITY ANALYSIS (k=20)")
    print("="*70)

    df_k20 = selectivity_df[selectivity_df['k'] == 20]
    df_k20 = df_k20[df_k20['method'].isin(methods)]

    print("\n--- Recall@20 by Selectivity ---")
    pivot = df_k20.pivot_table(
        index='selectivity_bin',
        columns='method',
        values='avg_recall@20'
    )
    print(pivot.to_string())

    print("\n--- P50 Latency (ms) by Selectivity ---")
    pivot_latency = df_k20.pivot_table(
        index='selectivity_bin',
        columns='method',
        values='p50_latency'
    )
    print(pivot_latency.to_string())

    print("\n" + "="*70)


def main():
    print("="*70)
    print("EXPERIMENTAL RESULTS ANALYSIS")
    print("="*70)

    comparison_df = pd.read_csv('results/comparison.csv')
    selectivity_df = pd.read_csv('results/selectivity_analysis.csv')

    print_summary(comparison_df, selectivity_df)

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    # Experiment 1
    plot_exp1_recall(comparison_df)
    plot_exp1_p50_latency(comparison_df)
    plot_exp1_p95_latency(comparison_df)
    plot_exp1_mean_latency(comparison_df)

    # Experiment 2
    plot_exp2_recall_vs_selectivity(selectivity_df)
    plot_exp2_latency_vs_selectivity(selectivity_df)
    plot_exp2_query_distribution(selectivity_df)

    print("\n" + "="*70)
    print("Analysis complete! Check results/figures/ for visualizations.")
    print("="*70)


if __name__ == '__main__':
    main()
