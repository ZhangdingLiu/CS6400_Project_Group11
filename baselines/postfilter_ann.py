"""
Post-Filter ANN Baseline

Owner: Zhangding Liu
Functionality: Baseline B - ANN search first, then filter
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


class PostFilterANN:
    """Baseline B: ANN search first, then filter"""

    def __init__(self, index, metadata_df: pd.DataFrame, fixed_nprobe: int = 32):
        """
        Initialize baseline.

        Args:
            index: IVFPQIndex instance
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
        raise NotImplementedError
