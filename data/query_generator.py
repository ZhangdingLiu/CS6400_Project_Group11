
"""
Query Workload Generator

Owner: Yao-Ting Huang
Functionality: Generate hybrid query workload
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _apply_filter(df: pd.DataFrame, f: Dict) -> pd.Series:
    """Return boolean mask for rows matching filter_dict."""
    mask = pd.Series(True, index=df.index)
    for col, spec in f.items():
        op = spec.get("op", "").upper()
        if op == "IN":
            mask &= df[col].isin(spec["values"])
        elif op == "EQ":
            mask &= df[col] == spec["value"]
        elif op == "RANGE":
            lo = spec.get("min", -np.inf)
            hi = spec.get("max", np.inf)
            mask &= (df[col] >= lo) & (df[col] <= hi)
        elif op in ("GT", "GTE", "LT", "LTE"):
            v = spec["value"]
            if op == "GT":
                mask &= df[col] > v
            elif op == "GTE":
                mask &= df[col] >= v
            elif op == "LT":
                mask &= df[col] < v
            else:
                mask &= df[col] <= v
        elif op == "NOT_IN":
            mask &= ~df[col].isin(spec["values"])
        else:
            raise ValueError(f"Unsupported op: {op} for column {col}")
    return mask


class QueryWorkloadGenerator:
    """Generate hybrid query workload with varying selectivity."""

    def __init__(self, metadata_df: pd.DataFrame, embeddings: np.ndarray, seed: int = 123) -> None:
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.embeddings = embeddings.astype(np.float32)
        self.N, self.d = self.embeddings.shape
        self.rng = np.random.default_rng(seed)

    def generate_query_embedding(self) -> np.ndarray:
        # Picks 3 random rows and forms a convex combination of those 3 vectors.
        idx = self.rng.choice(self.N, size=3, replace=False)
        w = self.rng.random(3).astype(np.float32)
        w = w / (w.sum() + 1e-8)
        q = (w[0] * self.embeddings[idx[0]] +
             w[1] * self.embeddings[idx[1]] +
             w[2] * self.embeddings[idx[2]])
        q = q + 0.01 * self.rng.normal(size=self.d).astype(np.float32)
        return _l2_normalize(q)

    def _pick_categories_by_mass(self, target_mass: float) -> List[int]:
        counts = self.metadata_df["category"].value_counts(normalize=True).sort_values(ascending=False)
        cats = []
        acc = 0.0
        for c, p in counts.items():
            cats.append(int(c))
            acc += float(p)
            if acc >= target_mass or len(cats) >= 5:
                break
        return cats

    def generate_filter(self, target_selectivity: float) -> Dict:
        t = float(target_selectivity)
        md = self.metadata_df

        def q(col, p):
            return int(np.quantile(md[col].to_numpy(), p, method="linear"))

        f: Dict = {}
        t = float(np.clip(target_selectivity, 1e-6, 0.99))
        
        md = self.metadata_df
        rng = self.rng


        # --- Decide how many filters to include ---
        if t <= 0.05:
            k = 3
        elif t <= 0.30:
            k = 2
        else:
            k = 1

        candidates: Dict[str, Dict] = {}

        # ---------------- YEAR ----------------
        # Randomly pick an op; bias RANGE for low/medium t, allow thresholds otherwise.
        if rng.random() < (0.75 if t <= 0.30 else 0.45):
            # RANGE: pick a random quantile window; narrower for small t
            if t <= 0.05:
                lo_q = float(rng.uniform(0.75, 0.90))
                width = float(rng.uniform(0.05, 0.12))
            elif t <= 0.30:
                lo_q = float(rng.uniform(0.45, 0.70))
                width = float(rng.uniform(0.12, 0.30))
            else:
                lo_q = float(rng.uniform(0.25, 0.65))
                width = float(rng.uniform(0.20, 0.45))
            hi_q = float(np.clip(lo_q + width, 0.0, 0.995))
            yr_lo, yr_hi = q("year", lo_q), q("year", hi_q)
            if yr_hi < yr_lo:
                yr_lo, yr_hi = yr_hi, yr_lo
            candidates["year"] = {"op": "RANGE", "min": yr_lo, "max": yr_hi}
        else:
            # Thresholded year (GTE or LT). Map t to quantile crudely.
            if rng.random() < 0.6:
                pth = float(np.clip(1.0 - 0.8 * t, 0.15, 0.95))  # smaller t -> higher threshold
                thr = q("year", pth)
                candidates["year"] = {"op": "GTE", "value": thr}
            else:
                pth = float(np.clip(0.25 + 0.5 * t, 0.05, 0.85))  # larger t -> higher cutoff for LT
                thr = q("year", pth)
                candidates["year"] = {"op": "LT", "value": thr}

        # IMPORTANCE: randomize op among GTE/GT/LT/EQ.
        p_imp = float(np.clip(1.0 - t, 0.05, 0.95))  # small t -> high quantile (stricter)
        imp_thr = q("importance", p_imp)
        op_choice = rng.choice(["GTE", "GT", "LT", "EQ"], p=[0.6, 0.15, 0.2, 0.05])
        if op_choice in ("GTE", "GT"):
            candidates["importance"] = {"op": op_choice, "value": imp_thr}
        elif op_choice == "LT":
            # flip to a looser lower-tail cutoff
            p_low = float(np.clip(0.15 + 0.5 * t, 0.05, 0.85))
            candidates["importance"] = {"op": "LT", "value": q("importance", p_low)}
        else:
            # EQ to a nearby/binned value; good enough as a “rough” equality
            val = imp_thr
            if pd.api.types.is_integer_dtype(md["importance"].dtype):
                val = int(val)
            candidates["importance"] = {"op": "EQ", "value": val}

        # REGION: small t -> EQ to a single region; medium: IN of 2; high: maybe skip or 2–3.
        regions_all = ["NA", "EU", "APAC", "LATAM", "AFR"]
        if t <= 0.05:
            chosen = rng.choice(regions_all)
            candidates["region"] = {"op": "EQ", "value": chosen}
        elif t <= 0.30:
            rset = list(rng.choice(regions_all, size=2, replace=False))
            candidates["region"] = {"op": "IN", "values": rset}
        else:
            if rng.random() < 0.5:
                # light filter
                rset = list(rng.choice(regions_all, size=int(rng.integers(2, 4)), replace=False))
                candidates["region"] = {"op": "IN", "values": rset}
            else:
                # sometimes skip region entirely by not adding it to candidates
                pass

        # CATEGORY: small t -> few cats
        if t <= 0.05:
            cats = self._pick_categories_by_mass(0.05)
            if rng.random() < 0.25 and len(cats) > 0:
                # occasionally EQ a single category
                candidates["category"] = {"op": "EQ", "value": cats[0]}
            else:
                candidates["category"] = {"op": "IN", "values": cats}
        elif t <= 0.30:
            candidates["category"] = {"op": "IN", "values": self._pick_categories_by_mass(0.15)}
        else:
            if rng.random() < 0.6:
                candidates["category"] = {"op": "IN", "values": self._pick_categories_by_mass(0.35)}
            else:
                # sometimes omit category entirely
                pass

        # Randomly pick some candidates
        keys = list(candidates.keys())
        if len(keys) == 0:
            return {"year": {"op": "GTE", "value": q("year", 0.30)}}

        keep_k = min(k, len(keys))
        picked = list(rng.choice(keys, size=keep_k, replace=False))
        return {k: candidates[k] for k in picked}


    def estimate_selectivity(self, filter_dict: Dict) -> float:
        mask = _apply_filter(self.metadata_df, filter_dict)
        return float(mask.mean())

    def _target_selectivities(self, n_queries: int, mode: str) -> np.ndarray:
        mode = mode.lower()
        if mode == "uniform":
            return self.rng.uniform(0.03, 0.8, size=n_queries)
        if mode == "low":
            return self.rng.uniform(0.03, 0.1, size=n_queries)
        if mode == "medium":
            return self.rng.uniform(0.1, 0.3, size=n_queries)
        if mode == "high":
            return self.rng.uniform(0.3, 0.8, size=n_queries)
        raise ValueError("selectivity_distribution must be one of {'uniform','low','medium','high'}.")


def generate_queries(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_queries: int,
    selectivity_ranges: List[Tuple[float, float]],
    k_value: int = 10,
) -> List[Dict]:
    """
    Generate a query workload with explicit per-query selectivity ranges.
        - For each query i:
            * Choose (lo, hi) = selectivity_ranges[i % len(selectivity_ranges)].
            * Try up to a fixed number (64) of random filters.
            * Accept the first filter with selectivity lies in [lo, hi].
            * Otherwise, return the closest candidate.
    Args:
        embeddings:
            np.ndarray (N, d), float32. Item embeddings.
        metadata:
            pd.DataFrame with N rows, using the standard metadata schema.
        n_queries:
            Total number of queries to generate.
        selectivity_ranges:
            List of (lo, hi) tuples with 0.0 <= lo <= hi <= 1.0.
            Query i will target the range selectivity_ranges[i % len(selectivity_ranges)].
        k_value:
            The top-k value to be used by downstream search (stored per query).

    Returns:
        list[dict]
        {
            "vector": np.ndarray of shape (d,),  
            "filter": filter_dict,               
            "selectivity": float,              
            "k": int                             # requested top-k
        }
    """
    if not selectivity_ranges:
        raise ValueError("selectivity_ranges must be a non-empty list of (lo, hi) tuples.")

    gen = QueryWorkloadGenerator(metadata, embeddings)
    queries: List[Dict] = []

    max_tries_per_query = 64  

    for i in range(n_queries):
        lo, hi = selectivity_ranges[i % len(selectivity_ranges)]
        lo = float(lo)
        hi = float(hi)
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"Invalid selectivity range {(lo, hi)}; must satisfy 0.0 <= lo <= hi <= 1.0")

        best_q = None
        best_diff = float("inf")

        # Try multiple times to hit the desired selectivity band
        for _ in range(max_tries_per_query):
            # Pick a target inside the range (centered a bit towards the middle)
            target = float(gen.rng.uniform(lo, hi))
            qvec = gen.generate_query_embedding()
            f = gen.generate_filter(target)
            sel = gen.estimate_selectivity(f)  # empirical selectivity

            # Accept if selecitivty is inside the desired band
            if lo <= sel <= hi:
                best_q = {
                    "vector": qvec,          
                    "filter": f,
                    "selectivity": float(sel),
                    "k": k_value,
                }
                break

            # Otherwise, track the closest candidate so far
            diff = 0.0
            if sel < lo:
                diff = lo - sel
            elif sel > hi:
                diff = sel - hi

            if diff < best_diff:
                best_diff = diff
                best_q = {
                    "vector": qvec,
                    "filter": f,
                    "selectivity": float(sel),
                    "k": k_value,
                }

        # After max_tries return the closest one
        queries.append(best_q)

    return queries

