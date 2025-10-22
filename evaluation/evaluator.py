"""
Hybrid Search Evaluator

Owner: Zhangding Liu
Functionality: Run experiments and compare different methods
"""

from typing import List, Dict, Any
import pandas as pd


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

    def evaluate_method(self, search_engine, method_name: str) -> pd.DataFrame:
        """
        Evaluate a single search method.

        Args:
            search_engine: Search engine instance
            method_name: Name of the method

        Returns:
            pd.DataFrame: Per-query and aggregate metrics
        """
        raise NotImplementedError

    def compare_methods(self, methods: Dict[str, Any]) -> pd.DataFrame:
        """
        Compare multiple methods.

        Args:
            methods: {method_name: search_engine}

        Returns:
            pd.DataFrame: Comparison table
        """
        raise NotImplementedError

    def run_selectivity_analysis(self, methods: Dict[str, Any]) -> pd.DataFrame:
        """
        Group queries by selectivity and analyze.

        Args:
            methods: {method_name: search_engine}

        Returns:
            pd.DataFrame: Performance grouped by selectivity bins
        """
        raise NotImplementedError
