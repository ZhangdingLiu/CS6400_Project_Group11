
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
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (N, d).")
        if len(metadata_df) != embeddings.shape[0]:
            raise ValueError("metadata and embeddings must have the same number of rows.")
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.embeddings = embeddings.astype(np.float32)
        self.N, self.d = self.embeddings.shape
        self.rng = np.random.default_rng(seed)

    def generate_query_embedding(self) -> np.ndarray:
        # Picks 3 random rows, draws random nonnegative weights that sum to 1, 
        # and forms a convex combo of those 3 embedding vectors.
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


        # --- Decide how many filters to include (fewer for high t) ---
        if t <= 0.05:
            k_min, k_max = 3, 4     # 3–4 filters
            k = 3
        elif t <= 0.30:
            k_min, k_max = 2, 3     # 2–3 filters
            k = 2
        else:
            k = 1
            #k_min, k_max = 1, 2     # 1–2 filters
        #k = int(rng.integers(k_min, k_max + 1))

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

        # -------------- IMPORTANCE --------------
        # Tie to t via a tail quantile; randomize op among GTE/GT/LT/EQ.
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

        # ---------------- REGION ----------------
        # For small t prefer EQ to a single region; medium: IN of 2; high: maybe skip or 2–3.
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

        # --------------- CATEGORY ---------------
        # Pick top-mass categories roughly proportional to t (small t -> few cats).
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

        # --- Randomly choose a subset of the built candidates ---
        keys = list(candidates.keys())
        if len(keys) == 0:
            # Fallback: always provide at least a year predicate
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

    def generate_workload(self, n_queries: int, selectivity_distribution: str = "uniform") -> List[Dict]:
        targets = self._target_selectivities(n_queries, selectivity_distribution)
        out: List[Dict] = []
        diff = []
        for t in targets:
            qvec = self.generate_query_embedding()
            f = self.generate_filter(float(t))
            sel = self.estimate_selectivity(f)
            out.append({
                "vector": qvec.tolist(),      # <-- convert to list
                "filter": f,
                "selectivity": float(sel),
            })
            diff.append(abs(t - sel))
            # print(t, sel)
        # optional debug:
        print("avg", sum(diff)/len(diff))
        return out

def generate_queries(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_queries: int,
    selectivity_ranges: List[Tuple[float, float]],
    k_value: int = 10,
) -> List[Dict]:
    """
    Generate a query workload with explicit per-query selectivity ranges.

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

    Generation behavior:
        - For each query i:
            * Choose (lo, hi) = selectivity_ranges[i % len(selectivity_ranges)].
            * Try up to a fixed number of random filters (and query vectors)
              using QueryWorkloadGenerator.generate_filter().
            * Accept the first filter whose empirical selectivity lies in [lo, hi].
            * If no exact hit is found within the budget, use the closest candidate
              in terms of selectivity distance to [lo, hi].

    Returns:
        list[dict], where each dict has the form:
        {
            "vector": np.ndarray of shape (d,),   # query embedding (L2-normalized)
            "filter": filter_dict,               # structured filter
            "selectivity": float,                # empirical fraction in [0, 1]
            "k": int                             # requested top-k
        }
    """
    if not selectivity_ranges:
        raise ValueError("selectivity_ranges must be a non-empty list of (lo, hi) tuples.")

    gen = QueryWorkloadGenerator(metadata, embeddings)
    queries: List[Dict] = []

    max_tries_per_query = 64  # cap to avoid infinite loops

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

            # If we are inside the desired band, accept immediately
            if lo <= sel <= hi:
                best_q = {
                    "vector": qvec,          # keep as np.ndarray; caller can .tolist() if needed
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

        # After max_tries_per_query, we either have an in-range query or the closest one we saw
        if best_q is None:
            raise RuntimeError("Failed to generate any query (this should not happen).")
        queries.append(best_q)

    return queries


if __name__ == "__main__":
    # Small synthetic smoke test
    N, d = 200000, 384
    emb = np.random.randn(N, d).astype(np.float32)
    md = pd.DataFrame({
        "year": np.random.randint(0, 101, size=N),
        "importance": np.random.randint(1, 101, size=N),
        "region": np.random.choice(
            ["NA", "EU", "APAC", "LATAM", "AFR"],
            size=N,
            p=[0.4, 0.25, 0.2, 0.1, 0.05],
        ),
        "category": np.random.randint(1, 31, size=N),
        "paragraph_len": np.random.randint(5, 400, size=N),
    })

    # Example selectivity ranges: low, medium, high
    sel_ranges = [
        (0.01, 0.05),
        (0.05, 0.15),
        (0.15, 0.40),
    ]
    import time

    t0 = time.perf_counter()
    qs = generate_queries(emb, md, n_queries=1000, selectivity_ranges=sel_ranges, k_value=10)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"[synthetic] generated {len(qs)} queries in {elapsed:.3f}s "
          f"({elapsed / len(qs):.4f}s/query)")
    #print(qs)
    print("Generated", len(qs), "queries.")
    print("Observed selectivities:", [round(q["selectivity"], 4) for q in qs])