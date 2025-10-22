"""
Metadata Signature Builder

Owner: Zaowei Dai
Functionality: Build metadata signatures for each IVF list
"""

from typing import Dict, Set
import numpy as np
import pandas as pd


class MetadataSignatureBuilder:
    """Build per-list metadata signatures"""

    def __init__(self, metadata_df: pd.DataFrame, list_assignments: np.ndarray):
        """
        Initialize signature builder.

        Args:
            metadata_df: Metadata DataFrame
            list_assignments: Shape (N,), IVF list assignments
        """
        self.metadata_df = metadata_df
        self.list_assignments = list_assignments

    def build_categorical_signatures(self, column: str) -> Dict[int, Set]:
        """
        Build signatures for categorical columns.

        Args:
            column: Column name

        Returns:
            Dict[int, Set]: {list_id: set_of_values}
        """
        raise NotImplementedError

    def build_numeric_signatures(self, column: str, n_buckets: int = 64) -> Dict:
        """
        Build signatures for numeric columns.

        Args:
            column: Column name
            n_buckets: Number of buckets

        Returns:
            Dict: Contains min_max, buckets, and global_buckets
        """
        raise NotImplementedError

    def build_value_counts(self, column: str) -> Dict[int, Dict]:
        """
        Build value counts for NOT IN predicates.

        Args:
            column: Column name

        Returns:
            Dict[int, Dict]: {list_id: {value: count}}
        """
        raise NotImplementedError

    def build_all_signatures(self) -> Dict:
        """
        Build signatures for all metadata columns.

        Returns:
            Dict: Complete signature dictionary
        """
        raise NotImplementedError
