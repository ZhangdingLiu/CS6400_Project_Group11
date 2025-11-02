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
def load_embeddings(dataset_name: str, n_samples: int) -> tuple:
    """
    Load data from HuggingFace

    Returns:
        embeddings: np.ndarray (N, d), float32, L2-normalized
        texts: list[str], length N
    """
```

#### `data/preprocessor.py`
```python
def generate_metadata(n_samples: int, config: dict) -> pd.DataFrame:
    """
    Generate synthetic metadata

    Returns:
        DataFrame with 5 columns (category, importance, year, paragraph_len, region)
    """
```

#### `data/query_generator.py`
```python
def generate_queries(embeddings, metadata, n_queries: int,
                     selectivity_ranges: list) -> list[dict]:
    """
    Generate query workload

    Returns:
        List of query dicts:
        {
            'vector': np.ndarray (d,),
            'filter': filter_dict,
            'selectivity': float
        }
    """
```

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
