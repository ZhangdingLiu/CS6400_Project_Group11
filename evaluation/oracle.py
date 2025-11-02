"""
Exact Search Oracle

Owner: Zhangding Liu
Functionality: Exact filtered search for computing recall ground truth
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.filter_utils import apply_filter
from utils.distance import batch_cosine_similarity


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

    def search(self, query: np.ndarray, filter_dict: Dict, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact filtered search.

        Args:
            query: Shape (d,), query vector
            filter_dict: Filter conditions
            k: Number of results

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, ids)
        """
        candidate_indices = apply_filter(self.metadata_df, filter_dict)

        if len(candidate_indices) == 0:
            return np.array([]), np.array([])

        candidate_embeddings = self.embeddings[candidate_indices]
        similarities = batch_cosine_similarity(query, candidate_embeddings)

        k = min(k, len(candidate_indices))
        top_k_local = np.argpartition(similarities, -k)[-k:]
        top_k_local = top_k_local[np.argsort(-similarities[top_k_local])]

        result_ids = candidate_indices[top_k_local]
        result_distances = similarities[top_k_local]

        return result_distances, result_ids
