"""
Hybrid Search Evaluator

Owner: Zhangding Liu
Functionality: Run experiments and compare different methods
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import time
from .metrics import recall_at_k, compute_latency_stats


class HybridSearchEvaluator:
    """Run experiments and compare methods"""

    def __init__(self, queries: List[Dict], oracle):
        """
        Initialize evaluator.

        Args:
            queries: List of query dictionaries
            oracle: ExactSearchOracle instance
        """
        self.queries = queries
        self.oracle = oracle

    def evaluate_method(self, search_engine, method_name: str, k_values: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Evaluate a single search method.

        Args:
            search_engine: Search engine instance
            method_name: Name of the method
            k_values: List of k values for Recall@k computation

        Returns:
            pd.DataFrame: Per-query metrics
        """
        results = []

        for query_idx, query_dict in enumerate(self.queries):
            # Extract query components
            query_vector = query_dict['vector']
            filter_dict = query_dict.get('filter', {})

            # Get results for each k value
            for k in k_values:
                # Get ground truth from oracle
                gt_distances, gt_ids = self.oracle.search(query_vector, filter_dict, k)

                # Measure latency and get method results
                start_time = time.perf_counter()
                try:
                    method_distances, method_ids = search_engine.search(query_vector, filter_dict, k)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Compute recall
                    if len(gt_ids) > 0:
                        recall = recall_at_k(method_ids, gt_ids)
                    else:
                        recall = 0.0

                except Exception as e:
                    print(f"Error in {method_name} for query {query_idx}: {e}")
                    latency_ms = 0.0
                    recall = 0.0

                results.append({
                    'query_id': query_idx,
                    'method': method_name,
                    'k': k,
                    f'recall@{k}': recall,
                    'latency_ms': latency_ms
                })

        return pd.DataFrame(results)

    def compare_methods(self, methods: Dict[str, Any], k_values: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Compare multiple methods.

        Args:
            methods: {method_name: search_engine}
            k_values: List of k values for Recall@k computation

        Returns:
            pd.DataFrame: Comparison table with aggregate statistics
        """
        all_results = []

        # Evaluate each method
        for method_name, search_engine in methods.items():
            print(f"Evaluating {method_name}...")
            method_df = self.evaluate_method(search_engine, method_name, k_values)
            all_results.append(method_df)

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)

        # Compute aggregate statistics per method and k
        comparison_results = []

        for method_name in methods.keys():
            method_data = combined_df[combined_df['method'] == method_name]

            for k in k_values:
                k_data = method_data[method_data['k'] == k]

                if len(k_data) > 0:
                    recall_col = f'recall@{k}'
                    latencies = k_data['latency_ms'].tolist()
                    latency_stats = compute_latency_stats(latencies)

                    comparison_results.append({
                        'method': method_name,
                        'k': k,
                        f'avg_recall@{k}': k_data[recall_col].mean(),
                        'p50_latency': latency_stats['p50'],
                        'p95_latency': latency_stats['p95'],
                        'p99_latency': latency_stats['p99'],
                        'mean_latency': latency_stats['mean'],
                        'n_queries': len(k_data)
                    })

        return pd.DataFrame(comparison_results)

    def run_selectivity_analysis(self, methods: Dict[str, Any]) -> pd.DataFrame:
        """
        Group queries by selectivity and analyze.

        Args:
            methods: {method_name: search_engine}

        Returns:
            pd.DataFrame: Performance grouped by selectivity bins
        """
        raise NotImplementedError
