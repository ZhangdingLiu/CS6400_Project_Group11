"""
Post-Filter ANN Baseline

Owner: Zhangding Liu
Functionality: Baseline B - ANN search first, then filter
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.filter_utils import apply_filter


class PostFilterANN:
    """Baseline B: ANN search first, then filter"""

    def __init__(self, index, metadata_df: pd.DataFrame, fixed_nprobe: int = 32):
        """
        Initialize baseline.

        Args:
            index: IVFPQIndex instance (or MockIVFPQIndex for testing)
            metadata_df: Metadata DataFrame
            fixed_nprobe: Fixed nprobe value
        """
        self.index = index
        self.metadata_df = metadata_df
        self.fixed_nprobe = fixed_nprobe

    def search(self, query: np.ndarray, filter_dict: Dict, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with ANN then filtering.

        Args:
            query: Shape (d,), normalized query vector
            filter_dict: Filter conditions
            k: Number of results to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, ids)
        """
        k_search = k * 2
        max_search = min(10000, len(self.metadata_df))

        while k_search <= max_search:
            distances, ids = self.index.search(query, k_search)

            if len(ids) == 0:
                return np.array([]), np.array([])

            valid_indices = apply_filter(self.metadata_df.iloc[ids], filter_dict)

            if len(valid_indices) >= k:
                valid_indices = valid_indices[:k]
                result_ids = ids[valid_indices]
                result_distances = distances[valid_indices]
                return result_distances, result_ids

            if k_search >= max_search:
                result_ids = ids[valid_indices]
                result_distances = distances[valid_indices]
                return result_distances, result_ids

            k_search = min(k_search * 2, max_search)

        return np.array([]), np.array([])
