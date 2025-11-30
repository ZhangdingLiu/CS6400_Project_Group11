"""
Metadata Signature Builder

Owner: Zaowei Dai
Functionality: Build metadata signatures for each IVF list
"""

from typing import Dict, Set, Any
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
        if len(metadata_df) != len(list_assignments):
            raise ValueError(
                f"metadata_df has {len(metadata_df)} rows but list_assignments has {len(list_assignments)} elements."
            )

        self.metadata_df = metadata_df.reset_index(drop=True)
        self.list_assignments = np.asarray(list_assignments, dtype=np.int32)
        self.list_ids = np.unique(self.list_assignments)

    # Internal helper to iterate per-list groups
    def _iter_list_groups(self):
        for lid in self.list_ids:
            mask = self.list_assignments == lid
            df_list = self.metadata_df[mask]
            yield int(lid), df_list

    def build_categorical_signatures(self, column: str) -> Dict[int, Set[Any]]:
        """
        Build signatures for categorical columns.

        Args:
            column: Column name

        Returns:
            Dict[int, Set]: {list_id: set_of_values}
        """
        if column not in self.metadata_df.columns:
            raise KeyError(f"Column '{column}' not found in metadata_df.")

        signatures: Dict[int, Set[Any]] = {}

        for lid, df_list in self._iter_list_groups():
            values = df_list[column].dropna().unique().tolist()
            signatures[lid] = set(values)

        return signatures

    def build_numeric_signatures(self, column: str, n_buckets: int = 64) -> Dict[str, Any]:
        """
        Build signatures for numeric columns.

        Args:
            column: Column name
            n_buckets: Number of buckets

        Returns:
            Dict: {
                "min_max": {list_id: (min, max)},
                "buckets": {list_id: set(bucket_indices)},
                "global_buckets": np.ndarray of shape (n_buckets+1,)
            }
        """
        if column not in self.metadata_df.columns:
            raise KeyError(f"Column '{column}' not found in metadata_df.")

        col = pd.to_numeric(self.metadata_df[column], errors="coerce")
        if col.isna().all():
            raise ValueError(f"Column '{column}' has no numeric values.")

        global_min = float(col.min())
        global_max = float(col.max())

        if global_min == global_max:
            edges = np.array([global_min, global_max], dtype=np.float32)
        else:
            edges = np.linspace(global_min, global_max, n_buckets + 1, dtype=np.float32)

        min_max: Dict[int, Any] = {}
        buckets: Dict[int, Set[int]] = {}

        for lid, df_list in self._iter_list_groups():
            vals = pd.to_numeric(df_list[column], errors="coerce").dropna()

            if vals.empty:
                min_max[lid] = (None, None)
                buckets[lid] = set()
                continue

            local_min = float(vals.min())
            local_max = float(vals.max())
            min_max[lid] = (local_min, local_max)

            idx = np.digitize(vals.to_numpy(), edges, right=False)
            idx = np.clip(idx, 1, len(edges) - 1)
            bucket_ids = set(int(i - 1) for i in np.unique(idx))
            buckets[lid] = bucket_ids

        return {
            "min_max": min_max,
            "buckets": buckets,
            "global_buckets": edges,
        }

    def build_value_counts(self, column: str) -> Dict[int, Dict[Any, int]]:
        """
        Build value counts for NOT IN predicates.

        Args:
            column: Column name

        Returns:
            Dict[int, Dict]: {list_id: {value: count}}
        """
        if column not in self.metadata_df.columns:
            raise KeyError(f"Column '{column}' not found in metadata_df.")

        result: Dict[int, Dict[Any, int]] = {}

        for lid, df_list in self._iter_list_groups():
            vc = df_list[column].value_counts(dropna=False)

            counts: Dict[Any, int] = {}
            for value, count in vc.items():
                key = value
                if isinstance(value, float) and np.isnan(value):
                    key = "NaN"
                counts[key] = int(count)

            result[lid] = counts

        return result

    def build_all_signatures(self) -> Dict[str, Any]:
        """
        Build signatures for all metadata columns.

        Returns:
            Dict: Complete signature dictionary
        """
        signatures: Dict[str, Any] = {
            "categorical": {},
            "numeric": {},
            "value_counts": {},
        }

        for col in self.metadata_df.columns:
            signatures["value_counts"][col] = self.build_value_counts(col)

            dtype = self.metadata_df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                signatures["numeric"][col] = self.build_numeric_signatures(col)
            else:
                signatures["categorical"][col] = self.build_categorical_signatures(col)

        return signatures
