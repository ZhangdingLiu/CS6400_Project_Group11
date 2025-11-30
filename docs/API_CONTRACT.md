# API Contract

This document defines the interface specifications between modules to ensure correct integration of code implemented by team members.

## Data Format Specifications

### Filter Dictionary
All modules must use the following format to represent filter conditions:
```python
filter_dict = {
    'category': {'op': 'IN', 'values': [1, 5, 10]},
    'year': {'op': 'RANGE', 'min': 20, 'max': 80},
    'importance': {'op': 'GT', 'value': 50},
    'region': {'op': 'EQ', 'value': 'EU'}
}
```

Supported operators:
- `IN`: Value is in the list
- `RANGE`: Range query (min, max)
- `GT/LT/GTE/LTE`: Greater than / Less than / Greater than or equal / Less than or equal
- `EQ`: Equal to

### Metadata DataFrame
5 fixed columns (row-aligned with embeddings):
- `category`: int32, [1, 30]
- `importance`: int32, [1, 100]
- `year`: int32, [0, 100]
- `paragraph_len`: int32
- `region`: str, {NA, EU, APAC, LATAM, AFR}

---

## Module Interfaces

### 1. Data Module (Yao-Ting Huang)

#### `data/loader.py`
```python
class WikipediaDataLoader:
    """
    Download & parse Wikipedia embedding dataset from HuggingFace.

    Supported embedding_method values:
        - "OpenAI"    -> wiki_openai.ndjson.gz, column "text-embedding-ada-002" (d=1536)
        - "MiniLM"    -> wiki_minilm.ndjson.gz, column "all-MiniLM-L6-v2"      (d=384)
        - "GTE-small" -> wiki_gte.ndjson.gz,    column "embedding"             (d=384)
    """

    def __init__(self, embedding_method: str = "OpenAI", text_field: str = "body"):
        """
        Args:
            embedding_method: Which embedding variant to use.
            text_field: Preferred text column name; will fall back to ['body', 'text', 'content'].
        """

    def load(self, n_samples: Optional[int] = None) -> tuple[list[str], np.memmap]:
        """
        Download the NDJSON file from HuggingFace, stream and parse lines,
        and return texts and embeddings.

        Args:
            n_samples:
                Maximum number of samples to load. If None, load up to the
                full dataset size (224,482 rows for the full Wikipedia dataset).

        Returns:
            texts:
                list[str] of length N' (N' <= n_samples or dataset size).
            embeddings:
                np.memmap of shape (N', d), dtype=float32.
                These are the **raw, unnormalized** embedding vectors.
        """
```

#### `data/preprocessor.py`
```python
class MetadataGenerator:
    """
    Generate synthetic metadata attributes for each paragraph.
    Uses a fixed RNG seed for reproducibility.
    """

    def build_metadata_table(self, n_samples: int, texts: list[str]) -> pd.DataFrame:
        """
        Build a complete metadata DataFrame aligned with the given texts.

        Args:
            n_samples: Number of samples (must equal len(texts)).
            texts: List of paragraph texts.

        Returns:
            pd.DataFrame with N rows and 5 columns:
                - category      (int32, [1, 30])
                - importance    (int32, [1, 100])
                - year          (int32, [0, 100])
                - paragraph_len (int32)
                - region        (string, {NA, EU, APAC, LATAM, AFR})
        """
```

#### `data/__init__.py`
```python
from .loader import WikipediaDataLoader
from .preprocessor import MetadataGenerator

def build_data_files(
    loader: WikipediaDataLoader,
    out_dir: str = "data_files",
    n_samples: Optional[int] = None,
) -> tuple[str, str]:
    """
    Materialize embeddings and metadata on disk.

    Steps:
        1) Calls `loader.load(n_samples)` to obtain:
               texts: list[str]
               X: np.memmap (N, d), float32, unnormalized embeddings
        2) Writes embeddings to {out_dir}/embeddings.npy using an on-disk memmap.
        3) Uses MetadataGenerator(seed=42) to build metadata and save to
           {out_dir}/metadata.parquet.

    Args:
        loader: Configured WikipediaDataLoader instance.
        out_dir: Output directory (created if missing).
        n_samples: Optional cap on number of rows to load.

    Creates:
        - {out_dir}/embeddings.npy    # float32, shape (N, d), unnormalized
        - {out_dir}/metadata.parquet  # metadata DataFrame with 5 columns

    Returns:
        (embeddings_path, metadata_path)
    """

```

#### Typical Usage to download embedding in .npy and generate metadata in .parquet (in `scripts/build_data.py`)
```python
from data import WikipediaDataLoader, build_data_files
def main():
    # embedding_method in {"OpenAI", "MiniLM", "GTE-small"}
    loader = WikipediaDataLoader(embedding_method="OpenAI", text_field="body")
    emb_path, meta_path = build_data_files(loader, out_dir="data_files")
    print("Saved:", emb_path, "and", meta_path)

if __name__ == "__main__":
    main()
```


#### `data/query_generator.py`
```python
def generate_queries(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_queries: int,
    selectivity_ranges: list[tuple[float, float]],
    k_value: int = 10,
) -> list[dict]:
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
```
#### `scripts/build_queries.py`
A simple script to generate query workload from the data files:


---

### 2. Indexing Module (Zaowei Dai)

#### `indexing/ivf_index.py`
```python
class IVFPQIndex:
    def __init__(self, d: int, nlist: int, m: int, nbits: int):
        """Initialize IVF-PQ index"""

    def train(self, embeddings: np.ndarray):
        """Train index"""

    def add(self, embeddings: np.ndarray):
        """Add vectors to index"""

    def get_list_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get IVF list ID for each vector

        Returns:
            np.ndarray (N,), dtype=int, value range [0, nlist-1]
        """

    def get_centroids(self) -> np.ndarray:
        """
        Returns:
            np.ndarray (nlist, d), centroid vectors
        """

    def search_preassigned(self, query: np.ndarray, list_ids: list[int],
                          k_prime: int) -> tuple:
        """
        Search only specified IVF lists

        Args:
            query: (d,) query vector
            list_ids: List IDs to search
            k_prime: Return top-k' results

        Returns:
            distances: np.ndarray (k_prime,)
            ids: np.ndarray (k_prime,), vector indices in dataset
        """

    def save(self, path: str):
        """Save index"""

    def load(self, path: str):
        """Load index"""
```

#### `indexing/metadata_signatures.py`
```python
class MetadataSignatureBuilder:
    def build_signatures(self, metadata: pd.DataFrame,
                        list_assignments: np.ndarray) -> dict:
        """
        Build metadata signatures for each IVF list

        Returns:
            {
                'category': {list_id: set_of_values},
                'importance': {
                    'min_max': {list_id: (min, max)},
                    'buckets': {list_id: bucket_mask},
                    'global_buckets': np.ndarray
                },
                'year': {...},
                'paragraph_len': {...},
                'region': {...}
            }
        """
```

---

### 3. Search Module (Yichang Xu)

#### `search/pruning.py`
```python
class FilterAwarePruner:
    def __init__(self, signatures: dict, centroids: np.ndarray):
        """Initialize pruner"""

    def apply_filter(self, filter_dict: dict) -> list[int]:
        """
        Prune IVF lists based on filter

        Returns:
            candidate_list_ids: List IDs that may contain results
        """

    def rank_lists_by_distance(self, query: np.ndarray,
                              candidate_lists: list[int]) -> list[int]:
        """
        Rank lists by query-to-centroid distance

        Returns:
            sorted_list_ids: Sorted from nearest to farthest
        """
```

#### `search/adaptive_deepening.py`
```python
class AdaptiveSearchPlanner:
    def initialize_parameters(self, k: int,
                            estimated_selectivity: float) -> tuple[int, int]:
        """
        Initialize nprobe and k_prime

        Returns:
            nprobe_0, k_prime_0
        """

    def should_deepen(self, n_results: int, k: int) -> bool:
        """Determine if deepening is needed"""

    def grow_parameters(self, nprobe: int, k_prime: int) -> tuple[int, int]:
        """
        Grow parameters

        Returns:
            new_nprobe, new_k_prime
        """
```

#### `search/search_engine.py`
```python
class HybridSearchEngine:
    def __init__(self, index: IVFPQIndex, metadata: pd.DataFrame,
                 signatures: dict, config: dict):
        """Initialize search engine"""

    def search(self, query: np.ndarray, filter_dict: dict,
              k: int) -> tuple:
        """
        Main hybrid search interface

        Args:
            query: (d,) normalized query vector
            filter_dict: Metadata filter conditions
            k: Return top-k results

        Returns:
            distances: np.ndarray (<=k,)
            ids: np.ndarray (<=k,), result IDs that satisfy filter
            stats: dict, contains {lists_probed, codes_scored, iterations}
        """
```

---

### 4. Baselines Module (Zhangding Liu)

#### `baselines/prefilter_bruteforce.py`
```python
class PreFilterBruteForce:
    def search(self, query: np.ndarray, filter_dict: dict,
              k: int) -> tuple:
        """
        Pre-filter baseline: filter first, then exact search

        Returns:
            distances, ids, stats
        """
```

#### `baselines/postfilter_ann.py`
```python
class PostFilterANN:
    def search(self, query: np.ndarray, filter_dict: dict,
              k: int) -> tuple:
        """
        Post-filter baseline: ANN search then filter, retry if insufficient

        Returns:
            distances, ids, stats
        """
```

---

### 5. Evaluation Module (Zhangding Liu)

#### `evaluation/oracle.py`
```python
class OracleSearch:
    def search(self, query: np.ndarray, filter_dict: dict,
              k: int) -> tuple:
        """
        Exact search ground truth

        Returns:
            distances, ids, stats
        """
```

#### `evaluation/metrics.py`
```python
def compute_recall(predicted_ids: np.ndarray,
                   ground_truth_ids: np.ndarray, k: int) -> float:
    """Compute recall@k"""

def compute_latency_stats(latencies: list[float]) -> dict:
    """
    Returns:
        {'p50': float, 'p95': float, 'p99': float, 'mean': float}
    """
```

#### `evaluation/evaluator.py`
```python
class Evaluator:
    def run_experiments(self, methods: dict, queries: list,
                       ground_truth: dict) -> pd.DataFrame:
        """
        Compare multiple methods

        Returns:
            DataFrame with columns: [method, selectivity_range, recall@k, p50, p95, p99]
        """
```

---

## Integration Checklist

Before merging code, ensure:

- [ ] Function signatures match API_CONTRACT exactly
- [ ] Return value types and formats conform to specifications
- [ ] Corresponding unit tests added
- [ ] filter_dict format is consistent
- [ ] metadata DataFrame column order is correct
- [ ] Vectors are L2-normalized (using cosine similarity)
