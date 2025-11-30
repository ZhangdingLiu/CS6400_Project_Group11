import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_experiment1_method_comparison(comparison_df, output_dir='results/figures'):
    """Experiment 1: Overall method comparison"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    k10 = comparison_df[comparison_df['k'] == 10].set_index('method')
    k20 = comparison_df[comparison_df['k'] == 20].set_index('method')

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    # Recall@10
    k10['avg_recall@10'].plot(kind='bar', ax=axes[0,0], color=colors)
    axes[0,0].set_title('Recall@10 by Method', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Recall@10')
    axes[0,0].set_ylim([0, 1.1])
    axes[0,0].grid(axis='y', alpha=0.3)
    axes[0,0].set_xlabel('')

    # Recall@20
    k20['avg_recall@20'].plot(kind='bar', ax=axes[0,1], color=colors)
    axes[0,1].set_title('Recall@20 by Method', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Recall@20')
    axes[0,1].set_ylim([0, 1.1])
    axes[0,1].grid(axis='y', alpha=0.3)
    axes[0,1].set_xlabel('')

    # P50 Latency
    latency_k10 = k10[['p50_latency']].rename(columns={'p50_latency': 'k=10'})
    latency_k20 = k20[['p50_latency']].rename(columns={'p50_latency': 'k=20'})
    latency_both = pd.concat([latency_k10, latency_k20], axis=1)
    latency_both.plot(kind='bar', ax=axes[1,0], color=['#3498db', '#e74c3c'])
    axes[1,0].set_title('P50 Latency by Method', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Latency (ms)')
    axes[1,0].grid(axis='y', alpha=0.3)
    axes[1,0].set_xlabel('')
    axes[1,0].legend()

    # P95 Latency
    p95_k10 = k10[['p95_latency']].rename(columns={'p95_latency': 'k=10'})
    p95_k20 = k20[['p95_latency']].rename(columns={'p95_latency': 'k=20'})
    p95_both = pd.concat([p95_k10, p95_k20], axis=1)
    p95_both.plot(kind='bar', ax=axes[1,1], color=['#3498db', '#e74c3c'])
    axes[1,1].set_title('P95 Latency by Method', fontsize=12, fontweight='bold')
    axes[1,1].set_ylabel('Latency (ms)')
    axes[1,1].grid(axis='y', alpha=0.3)
    axes[1,1].set_xlabel('')
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp1_method_comparison.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp1_method_comparison.png')
    plt.close()


def plot_experiment2_selectivity_analysis(selectivity_df, output_dir='results/figures'):
    """Experiment 2: Performance by selectivity (filter-aware value)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_k10 = selectivity_df[selectivity_df['k'] == 10]
    df_k20 = selectivity_df[selectivity_df['k'] == 20]

    bin_order = ['0.01-0.05', '0.05-0.15', '0.15-0.3', '0.3-1.0']
    bin_labels = ['Very Selective\n(1-5%)', 'Selective\n(5-15%)',
                  'Medium\n(15-30%)', 'High\n(>30%)']

    methods = ['Oracle', 'PreFilter', 'PostFilter', 'HybridSearch']
    colors_map = {'Oracle': '#2ecc71', 'PreFilter': '#3498db',
                  'PostFilter': '#e74c3c', 'HybridSearch': '#f39c12'}

    # Recall@10 vs Selectivity
    for method in methods:
        data = df_k10[df_k10['method'] == method]
        recalls = [data[data['selectivity_bin'] == b]['avg_recall@10'].values[0]
                  if len(data[data['selectivity_bin'] == b]) > 0 else np.nan
                  for b in bin_order]
        axes[0,0].plot(range(len(bin_order)), recalls, marker='o',
                      label=method, linewidth=2.5, markersize=8,
                      color=colors_map[method])

    axes[0,0].set_xticks(range(len(bin_order)))
    axes[0,0].set_xticklabels(bin_labels, fontsize=9)
    axes[0,0].set_ylabel('Recall@10', fontsize=11)
    axes[0,0].set_title('Recall@10 vs Selectivity', fontsize=12, fontweight='bold')
    axes[0,0].legend(loc='lower right')
    axes[0,0].grid(alpha=0.3)
    axes[0,0].set_ylim([0, 1.1])

    # Recall@20 vs Selectivity
    for method in methods:
        data = df_k20[df_k20['method'] == method]
        recalls = [data[data['selectivity_bin'] == b]['avg_recall@20'].values[0]
                  if len(data[data['selectivity_bin'] == b]) > 0 else np.nan
                  for b in bin_order]
        axes[0,1].plot(range(len(bin_order)), recalls, marker='o',
                      label=method, linewidth=2.5, markersize=8,
                      color=colors_map[method])

    axes[0,1].set_xticks(range(len(bin_order)))
    axes[0,1].set_xticklabels(bin_labels, fontsize=9)
    axes[0,1].set_ylabel('Recall@20', fontsize=11)
    axes[0,1].set_title('Recall@20 vs Selectivity', fontsize=12, fontweight='bold')
    axes[0,1].legend(loc='lower right')
    axes[0,1].grid(alpha=0.3)
    axes[0,1].set_ylim([0, 1.1])

    # P50 Latency vs Selectivity
    for method in methods:
        data = df_k10[df_k10['method'] == method]
        latencies = [data[data['selectivity_bin'] == b]['p50_latency'].values[0]
                    if len(data[data['selectivity_bin'] == b]) > 0 else np.nan
                    for b in bin_order]
        axes[1,0].plot(range(len(bin_order)), latencies, marker='o',
                      label=method, linewidth=2.5, markersize=8,
                      color=colors_map[method])

    axes[1,0].set_xticks(range(len(bin_order)))
    axes[1,0].set_xticklabels(bin_labels, fontsize=9)
    axes[1,0].set_ylabel('P50 Latency (ms)', fontsize=11)
    axes[1,0].set_title('P50 Latency vs Selectivity (k=10)', fontsize=12, fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)

    # Number of queries per bin
    bin_counts = df_k10.groupby('selectivity_bin')['n_queries'].first()
    bin_counts = [bin_counts[b] if b in bin_counts.index else 0 for b in bin_order]

    axes[1,1].bar(range(len(bin_order)), bin_counts, color='#95a5a6', alpha=0.7)
    axes[1,1].set_xticks(range(len(bin_order)))
    axes[1,1].set_xticklabels(bin_labels, fontsize=9)
    axes[1,1].set_ylabel('Number of Queries', fontsize=11)
    axes[1,1].set_title('Query Distribution by Selectivity', fontsize=12, fontweight='bold')
    axes[1,1].grid(axis='y', alpha=0.3)

    for i, count in enumerate(bin_counts):
        axes[1,1].text(i, count + 5, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/exp2_selectivity_analysis.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir}/exp2_selectivity_analysis.png')
    plt.close()


def plot_experiment3_memory_overhead(output_dir='results/figures'):
    """Experiment 3: Memory and build time overhead"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        stats_df = pd.read_csv('results/index_stats.csv')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Memory breakdown
        memory_data = {
            'Index': stats_df['index_size_mb'].values[0],
            'Signatures': stats_df['signature_size_mb'].values[0]
        }

        colors = ['#3498db', '#e74c3c']
        bars = ax1.bar(memory_data.keys(), memory_data.values(), color=colors, alpha=0.8)
        ax1.set_ylabel('Size (MB)', fontsize=11)
        ax1.set_title('Memory Overhead Breakdown', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, memory_data.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                    f'{val:.2f} MB', ha='center', fontsize=10)

        # Build time
        build_time = stats_df['index_build_time_seconds'].values[0]
        ax2.bar(['Index Build'], [build_time], color='#f39c12', alpha=0.8, width=0.4)
        ax2.set_ylabel('Time (seconds)', fontsize=11)
        ax2.set_title('Index Build Time', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.text(0, build_time + 0.5, f'{build_time:.2f}s', ha='center', fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/exp3_memory_overhead.png', dpi=150, bbox_inches='tight')
        print(f'Saved: {output_dir}/exp3_memory_overhead.png')
        plt.close()

    except Exception as e:
        print(f'Warning: Could not generate memory overhead plot: {e}')


def print_detailed_summary(comparison_df, selectivity_df):
    print("\n" + "="*70)
    print("EXPERIMENT 1: METHOD COMPARISON")
    print("="*70)

    print("\n--- Overall Performance (All Queries) ---")
    for k in [10, 20]:
        print(f"\nk = {k}:")
        cols = ['method', f'avg_recall@{k}', 'p50_latency', 'p95_latency', 'mean_latency']
        summary = comparison_df[comparison_df['k'] == k][cols].copy()
        summary.columns = ['Method', f'Recall@{k}', 'P50 (ms)', 'P95 (ms)', 'Mean (ms)']
        print(summary.to_string(index=False))

    print("\n" + "="*70)
    print("EXPERIMENT 2: SELECTIVITY ANALYSIS")
    print("="*70)

    print("\n--- Recall@10 by Selectivity Range ---")
    pivot = selectivity_df[selectivity_df['k'] == 10].pivot_table(
        index='selectivity_bin',
        columns='method',
        values='avg_recall@10'
    )
    print(pivot.to_string())

    print("\n--- P50 Latency (ms) by Selectivity Range ---")
    pivot_latency = selectivity_df[selectivity_df['k'] == 10].pivot_table(
        index='selectivity_bin',
        columns='method',
        values='p50_latency'
    )
    print(pivot_latency.to_string())

    print("\n--- Query Distribution ---")
    dist = selectivity_df[selectivity_df['k'] == 10].groupby('selectivity_bin')['n_queries'].first()
    print(dist.to_string())

    print("\n" + "="*70)
    print("EXPERIMENT 3: MEMORY & BUILD TIME")
    print("="*70)

    try:
        stats = pd.read_csv('results/index_stats.csv')
        print(f"\nIndex Build Time: {stats['index_build_time_seconds'].values[0]:.2f} seconds")
        print(f"Index Size: {stats['index_size_mb'].values[0]:.2f} MB")
        print(f"Signature Size: {stats['signature_size_mb'].values[0]:.2f} MB")
        print(f"Total Size: {stats['total_size_mb'].values[0]:.2f} MB")
    except:
        print("\nIndex stats not available")


def main():
    print("="*70)
    print("EXPERIMENTAL RESULTS ANALYSIS")
    print("="*70)

    comparison_df = pd.read_csv('results/comparison.csv')
    selectivity_df = pd.read_csv('results/selectivity_analysis.csv')

    print_detailed_summary(comparison_df, selectivity_df)

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    plot_experiment1_method_comparison(comparison_df)
    plot_experiment2_selectivity_analysis(selectivity_df)
    plot_experiment3_memory_overhead()

    print("\n" + "="*70)
    print("Analysis complete! Check results/figures/ for visualizations.")
    print("="*70)


if __name__ == '__main__':
    main()
