"""
Exact Search Oracle

Owner: Zhangding Liu
Functionality: Exact filtered search for computing recall ground truth
"""

from typing import Dict
import numpy as np
import pandas as pd


class ExactSearchOracle:
    """Exact filtered search (ground truth)"""

    def __init__(self, embeddings: np.ndarray, metadata_df: pd.DataFrame):
        """
        Initialize oracle.

        Args:
            embeddings: Shape (N, d), normalized embeddings
            metadata_df: Metadata DataFrame
        """
        self.embeddings = embeddings
        self.metadata_df = metadata_df

    def search(self, query: np.ndarray, filter_dict: Dict, k: int) -> np.ndarray:
        """
        Exact filtered search.

        Args:
            query: Shape (d,), query vector
            filter_dict: Filter conditions
            k: Number of results

        Returns:
            np.ndarray: Shape (k,), ground truth IDs
        """
        raise NotImplementedError
