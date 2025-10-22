"""
Pre-Filter + Brute-Force Baseline

Owner: Zhangding Liu
Functionality: Baseline A - Filter first, then exact search
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


class PreFilterBruteForce:
    """Baseline A: Filter first, then exact search"""

    def __init__(self, embeddings: np.ndarray, metadata_df: pd.DataFrame):
        """
        Initialize baseline.

        Args:
            embeddings: Shape (N, d), normalized embeddings
            metadata_df: Metadata DataFrame
        """
        self.embeddings = embeddings
        self.metadata_df = metadata_df

    def search(self, query: np.ndarray, filter_dict: Dict, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with pre-filtering and brute-force.

        Args:
            query: Shape (d,), normalized query vector
            filter_dict: Filter conditions
            k: Number of results to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, ids)
        """
        raise NotImplementedError
