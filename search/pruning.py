"""
Filter-Aware Pruner

Owner: Yichang Xu
Functionality: IVF list pruning based on metadata signatures
"""

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np


class FilterAwarePruner:
    """Prune IVF lists based on metadata signatures"""

    def __init__(self, signatures: Dict, centroids: np.ndarray):
        """
        Initialize pruner.

        Args:
            signatures: Metadata signatures
            centroids: Shape (nlist, d), IVF centroids
        """
        self.signatures = signatures or {}
        self.centroids = np.asarray(centroids) if centroids is not None else None

        self.numeric_signatures = {
            col: sig
            for col, sig in self.signatures.items()
            if isinstance(sig, dict) and ("min_max" in sig or "buckets" in sig or "global_buckets" in sig)
        }
        self.categorical_signatures = {
            col: sig
            for col, sig in self.signatures.items()
            if col not in self.numeric_signatures
        }

        self._all_list_ids = self._infer_list_ids()

    def _infer_list_ids(self) -> List[int]:
        ids: Set[int] = set()
        if self.centroids is not None and self.centroids.size > 0:
            ids.update(range(self.centroids.shape[0]))

        for sig in self.categorical_signatures.values():
            if isinstance(sig, dict):
                ids.update(int(i) for i in sig.keys())

        for sig in self.numeric_signatures.values():
            for subkey in ("min_max", "buckets", "value_counts"):
                subdict = sig.get(subkey)
                if isinstance(subdict, dict):
                    ids.update(int(i) for i in subdict.keys())

        if not ids and self.centroids is not None:
            ids.update(range(self.centroids.shape[0]))

        return sorted(ids)

    def _get_list_ids(self) -> List[int]:
        return self._all_list_ids or []

    def _mask_has_overlap(self, mask: object, indices: Iterable[int]) -> bool:
        indices = list(indices)
        if not indices:
            return True

        if mask is None:
            return True

        if isinstance(mask, (list, tuple, np.ndarray)):
            arr = np.asarray(mask, dtype=bool)
            for idx in indices:
                if 0 <= idx < arr.size and arr[idx]:
                    return True
            return False

        if isinstance(mask, set):
            return any(idx in mask for idx in indices)

        if isinstance(mask, int):
            for idx in indices:
                if idx < 0:
                    continue
                if mask & (1 << idx):
                    return True
            return False

        return True

    def _bucket_indices_for_range(self, buckets: Sequence[Sequence[float]], lo: float, hi: float) -> List[int]:
        if hi < lo:
            lo, hi = hi, lo

        indices: List[int] = []
        for idx, bounds in enumerate(buckets):
            if bounds is None:
                continue
            if isinstance(bounds, np.ndarray):
                if bounds.size < 2:
                    continue
                b_lo, b_hi = float(bounds[0]), float(bounds[1])
            elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                b_lo, b_hi = float(bounds[0]), float(bounds[1])
            else:
                continue

            if hi < b_lo or lo > b_hi:
                continue
            indices.append(idx)
        return indices

    def _list_passes_filter(self, list_id: int, filter_dict: Dict) -> bool:
        for column, spec in filter_dict.items():
            if not isinstance(spec, dict):
                continue

            op = str(spec.get("op", "")).upper()
            if not op:
                # Infer operation based on available keys.
                if "values" in spec:
                    op = "IN"
                elif {"min", "max"}.intersection(spec.keys()):
                    op = "RANGE"
                elif "value" in spec:
                    op = "EQ"
                else:
                    continue

            if op == "IN":
                values = spec.get("values")
                if not values:
                    return False
                value_set = set(values)
                if column in self.categorical_signatures:
                    if not self.test_categorical_in(list_id, column, value_set):
                        return False
                elif column in self.numeric_signatures:
                    match_found = any(
                        self.test_numeric_range(list_id, column, float(v), float(v))
                        for v in value_set
                    )
                    if not match_found:
                        return False
                else:
                    continue

            elif op == "EQ":
                if "value" not in spec:
                    continue
                value = spec["value"]
                if column in self.numeric_signatures:
                    if not self.test_numeric_range(list_id, column, float(value), float(value)):
                        return False
                else:
                    if not self.test_categorical_in(list_id, column, {value}):
                        return False

            elif op == "RANGE":
                lo = spec.get("min", spec.get("lo", spec.get("low", None)))
                hi = spec.get("max", spec.get("hi", spec.get("high", None)))
                if lo is None and hi is None:
                    continue
                lo = float(-np.inf if lo is None else lo)
                hi = float(np.inf if hi is None else hi)
                if not self.test_numeric_range(list_id, column, lo, hi):
                    return False

            elif op == "GTE":
                value = spec.get("value")
                if value is None:
                    continue
                if not self.test_numeric_range(list_id, column, float(value), float(np.inf)):
                    return False

            elif op == "GT":
                value = spec.get("value")
                if value is None:
                    continue
                lower = float(value)
                if not self.test_numeric_range(list_id, column, np.nextafter(lower, np.inf), float(np.inf)):
                    return False

            elif op == "LTE":
                value = spec.get("value")
                if value is None:
                    continue
                if not self.test_numeric_range(list_id, column, float(-np.inf), float(value)):
                    return False

            elif op == "LT":
                value = spec.get("value")
                if value is None:
                    continue
                upper = float(value)
                if not self.test_numeric_range(list_id, column, float(-np.inf), np.nextafter(upper, -np.inf)):
                    return False
            else:
                # Unsupported predicate types -> skip pruning for safety
                continue

        return True

    def test_categorical_in(self, list_id: int, column: str, values: Set) -> bool:
        """
        Test if list can contain any of the specified values.

        Args:
            list_id: IVF list ID
            column: Column name
            values: Set of allowed values

        Returns:
            bool: True if list might contain matching records
        """
        if not values:
            return False

        sig = self.categorical_signatures.get(column)
        if not isinstance(sig, dict):
            return True

        list_values = sig.get(list_id)
        if list_values is None:
            return True

        return bool(set(list_values) & set(values))

    def test_numeric_range(self, list_id: int, column: str, lo: float, hi: float) -> bool:
        """
        Test if list can contain values in range.

        Args:
            list_id: IVF list ID
            column: Column name
            lo: Lower bound
            hi: Upper bound

        Returns:
            bool: True if list might contain matching records
        """
        sig = self.numeric_signatures.get(column)
        if not isinstance(sig, dict):
            return True

        lo_val = float(lo)
        hi_val = float(hi)
        if hi_val < lo_val:
            lo_val, hi_val = hi_val, lo_val

        min_max = sig.get("min_max")
        if isinstance(min_max, dict):
            bounds = min_max.get(list_id)
            if bounds is not None:
                col_min, col_max = bounds
                if hi_val < float(col_min) or lo_val > float(col_max):
                    return False

        buckets = sig.get("buckets")
        global_buckets = sig.get("global_buckets")
        if isinstance(buckets, dict) and global_buckets is not None:
            mask = buckets.get(list_id)
            if mask is not None:
                idxs = self._bucket_indices_for_range(global_buckets, lo_val, hi_val)
                if idxs and not self._mask_has_overlap(mask, idxs):
                    return False

        return True

    def apply_filter(self, filter_dict: Dict) -> List[int]:
        """
        Apply filter conditions to all IVF lists.

        Args:
            filter_dict: Filter dictionary

        Returns:
            List[int]: IVF list IDs that pass all predicates
        """
        list_ids = self._get_list_ids()
        if not filter_dict or not isinstance(filter_dict, dict):
            return list(list_ids)

        results: List[int] = []
        for list_id in list_ids:
            if self._list_passes_filter(list_id, filter_dict):
                results.append(list_id)
        return results

    def rank_lists_by_distance(self, query: np.ndarray, candidate_lists: List[int]) -> List[int]:
        """
        Rank candidate lists by centroid distance.

        Args:
            query: Shape (d,), query vector
            candidate_lists: List IDs to rank

        Returns:
            List[int]: List IDs sorted by distance
        """
        if self.centroids is None or query is None or len(candidate_lists) == 0:
            return list(candidate_lists)

        candidate_lists = [int(i) for i in candidate_lists]
        nlist = self.centroids.shape[0]
        valid = [lid for lid in candidate_lists if 0 <= lid < nlist]
        invalid = [lid for lid in candidate_lists if lid < 0 or lid >= nlist]

        if not valid:
            return list(invalid)

        centroids_subset = self.centroids[valid]
        diff = centroids_subset - query.reshape(1, -1)
        dists = np.linalg.norm(diff, axis=1)
        order = np.argsort(dists, kind="stable")
        ranked = [valid[i] for i in order]
        return ranked + invalid
