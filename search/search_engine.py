"""
Hybrid Search Engine

Owner: Yichang Xu
Functionality: Main search coordinator integrating all components
"""

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from utils.filter_utils import apply_filter


class HybridSearchEngine:
    """Main search coordinator integrating all components"""

    def __init__(self, index, pruner, planner, metadata_df: pd.DataFrame):
        """
        Initialize search engine.

        Args:
            index: IVFPQIndex instance
            pruner: FilterAwarePruner instance
            planner: AdaptiveSearchPlanner instance
            metadata_df: Metadata DataFrame
        """
        self.index = index
        self.pruner = pruner
        self.planner = planner
        self.metadata_df = metadata_df
        self._metadata_size = int(len(metadata_df)) if metadata_df is not None else 0
        self.last_stats: Dict[str, int] = {}

    def _estimate_selectivity(self, filter_dict: Dict) -> float:
        if self.metadata_df is None or self._metadata_size == 0:
            return 1.0

        indices = apply_filter(self.metadata_df, filter_dict or {})
        estimate = float(len(indices)) / float(self._metadata_size) if self._metadata_size else 1.0
        lower_bound = 1.0 / max(self._metadata_size, 1)
        return max(lower_bound, estimate)

    def _apply_record_filter(
        self, ids: np.ndarray, distances: np.ndarray, filter_dict: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        if ids.size == 0:
            return np.empty(0, dtype=distances.dtype), np.empty(0, dtype=ids.dtype)

        valid_mask = ids >= 0
        if not np.any(valid_mask):
            return np.empty(0, dtype=distances.dtype), np.empty(0, dtype=ids.dtype)

        ids = ids[valid_mask]
        distances = distances[valid_mask]

        if not filter_dict or self.metadata_df is None or self._metadata_size == 0:
            return distances, ids

        subset = self.metadata_df.iloc[ids]
        match_idx = apply_filter(subset, filter_dict)
        if len(match_idx) == 0:
            return np.empty(0, dtype=distances.dtype), np.empty(0, dtype=ids.dtype)

        return distances[match_idx], ids[match_idx]

    def search(self, query: np.ndarray, filter_dict: Dict, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute hybrid search.

        Args:
            query: Shape (d,), normalized query vector
            filter_dict: Filter conditions
            k: Number of results to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, ids)
        """
        if query is None:
            raise ValueError("query must be provided.")
        if k <= 0:
            raise ValueError("k must be positive.")

        query = np.asarray(query, dtype=np.float32).reshape(-1)

        filter_dict = filter_dict or {}
        candidate_lists = self.pruner.apply_filter(filter_dict)
        if not candidate_lists:
            empty = np.empty(0, dtype=np.float32)
            empty_ids = np.empty(0, dtype=np.int64)
            self.last_stats = {"iterations": 0, "lists_probed": 0, "codes_scored": 0}
            return empty, empty_ids

        ordered_lists = list(self.pruner.rank_lists_by_distance(query, candidate_lists))
        lists_available = len(ordered_lists)
        estimated_selectivity = self._estimate_selectivity(filter_dict)
        nprobe, k_prime = self.planner.initialize_parameters(k, estimated_selectivity)
        nprobe = max(1, min(nprobe, lists_available))

        if self._metadata_size:
            k_prime = min(k_prime, self._metadata_size)

        collected_dists: List[float] = []
        collected_ids: List[int] = []
        seen_ids: Set[int] = set()
        stats = {"iterations": 0, "lists_probed": 0, "codes_scored": 0}

        while True:
            stats["iterations"] += 1
            probe_count = min(nprobe, lists_available)
            if probe_count == 0:
                break

            lists_to_probe = ordered_lists[:probe_count]
            stats["lists_probed"] = max(stats["lists_probed"], len(lists_to_probe))
            distances, ids = self.index.search_preassigned(query, lists_to_probe, k_prime)
            if not isinstance(distances, np.ndarray):
                distances = np.asarray(distances)
            if not isinstance(ids, np.ndarray):
                ids = np.asarray(ids)

            stats["codes_scored"] += int(len(ids))
            filtered_dists, filtered_ids = self._apply_record_filter(ids, distances, filter_dict)

            for dist, idx in zip(filtered_dists, filtered_ids):
                if idx in seen_ids:
                    continue
                seen_ids.add(int(idx))
                collected_dists.append(float(dist))
                collected_ids.append(int(idx))

            if len(collected_ids) >= k:
                break

            if not self.planner.should_deepen(len(collected_ids), k):
                break

            prev_nprobe, prev_k_prime = nprobe, k_prime
            nprobe, k_prime = self.planner.grow_parameters(nprobe, k_prime)
            nprobe = min(nprobe, lists_available)
            if self._metadata_size:
                k_prime = min(k_prime, self._metadata_size)

            if nprobe == prev_nprobe and k_prime == prev_k_prime:
                break

        self.last_stats = stats

        if not collected_ids:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        dists_array = np.asarray(collected_dists, dtype=np.float32)
        ids_array = np.asarray(collected_ids, dtype=np.int64)
        order = np.argsort(dists_array, kind="stable")
        order = order[:k]
        return dists_array[order], ids_array[order]
