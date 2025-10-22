# Task Assignment - CS6400 Hybrid Vector Search

## Quick Start

1. **Clone the repo**: `git clone <repo-url>`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Read your task below**
4. **Create your files** in the assigned folder
5. **Submit a Pull Request** when done

---

## Task 1: Data Module (Yao-Ting Huang)

**Your folder**: `data/`

**What you need to create**:
- `data/loader.py` - Load Wikipedia embeddings from HuggingFace
- `data/preprocessor.py` - Generate 5 metadata attributes
- `data/query_generator.py` - Generate query workload

**Outputs you must produce**:
- `data_files/embeddings.npy` - Shape (N, d), normalized vectors
- `data_files/metadata.parquet` - 5 columns: category, importance, year, paragraph_len, region
- `data_files/queries.json` - List of queries with filters

**Metadata specifications**:
1. **category**: int [1-30], Zipf distribution (s=1.2)
2. **importance**: int [1-100], Normal(mean=50, std=15)
3. **year**: int [0-100], Mixture distribution
4. **paragraph_len**: int, token count from text
5. **region**: string, {NA, EU, APAC, LATAM, AFR}, probabilities [0.4, 0.25, 0.2, 0.1, 0.05]

**Query format**:
```python
{
    "query_vec": np.ndarray,  # normalized
    "filter": {
        "category": {"op": "IN", "values": [1, 5, 10]},
        "year": {"op": "RANGE", "min": 20, "max": 80}
    },
    "k": 10,
    "selectivity": 0.15  # estimated
}
```


---

## Task 2: Indexing Module (Zaowei Dai) 

**Your folder**: `indexing/`

**What you need to create**:
- `indexing/ivf_index.py` - FAISS IVF-PQ wrapper
- `indexing/metadata_signatures.py` - Build per-list signatures
- `indexing/signature_storage.py` - Save/load signatures

**Key functions required**:

### IVFPQIndex class:
```python
def __init__(d, nlist, m=64, nbits=8)
def train(embeddings)
def add(embeddings)
def get_list_assignments(embeddings) -> np.ndarray  # Shape (N,)
def get_centroids() -> np.ndarray  # Shape (nlist, d)
def search_preassigned(query, list_ids, k) -> (distances, ids)
```

### MetadataSignatureBuilder class:
```python
def __init__(metadata_df, list_assignments)
def build_categorical_signatures(column) -> Dict[list_id, Set]
def build_numeric_signatures(column) -> Dict with min/max and buckets
def build_all_signatures() -> Dict
```

**Signature format**:
```python
{
    'category': {list_id: set([1, 5, 10, ...])},
    'importance': {
        'min_max': {list_id: (min, max)},
        'buckets': {list_id: 64-bit mask},
        'global_buckets': np.ndarray
    },
    # similar for year, paragraph_len, region
}
```

**Outputs**:
- `index_files/index.faiss`
- `index_files/signatures.parquet`

**Critical requirement**: NO FALSE NEGATIVES in signatures (never prune a list that contains valid results)


---

## Task 3: Search Module (Yichang Xu)

**Your folder**: `search/`

**What you need to create**:
- `search/pruning.py` - Filter-aware pruner
- `search/adaptive_deepening.py` - Adaptive parameter planner
- `search/search_engine.py` - Main search coordinator

**Key functions required**:

### FilterAwarePruner class:
```python
def __init__(signatures, centroids)
def apply_filter(filter_dict) -> List[int]  # Valid list IDs
def rank_lists_by_distance(query, candidate_lists) -> List[int]  # Sorted
```

### AdaptiveSearchPlanner class:
```python
def initialize_parameters(k, selectivity) -> (nprobe_0, k_prime_0)
def should_deepen(n_results, k) -> bool
def grow_parameters(nprobe, k_prime) -> (new_nprobe, new_k_prime)
```

### HybridSearchEngine class:
```python
def __init__(index, pruner, planner, metadata_df)
def search(query, filter_dict, k) -> (distances, ids)
```

**Search algorithm**:
1. Prune IVF lists using signatures
2. Rank lists by centroid distance
3. Initialize nprobe and k'
4. Loop:
   - Search top nprobe lists with k' candidates
   - Apply record-level filter
   - If results >= k: return top-k
   - Else: grow nprobe and k', retry

**Dependencies**: Requires Zaowei's signatures


---

## Task 4: Baselines & Evaluation (Zhangding Liu)

**Your folders**: `baselines/`, `evaluation/`, `experiments/`

**What you need to create**:

### Baselines:
- `baselines/prefilter_bruteforce.py` - Filter first, then exact search
- `baselines/postfilter_ann.py` - ANN search first, then filter

### Evaluation:
- `evaluation/oracle.py` - Exact filtered search (ground truth)
- `evaluation/metrics.py` - Recall@k, latency statistics
- `evaluation/evaluator.py` - Run experiments and compare methods

### Experiments:
- `experiments/run_experiments.py` - Main pipeline
- `experiments/analyze_results.py` - Generate charts

**All search methods must have same interface**:
```python
def search(query, filter_dict, k) -> (distances, ids)
```

**Metrics to compute**:
- Recall@k vs oracle
- Latency: P50, P95, P99
- By selectivity bins: [0.01, 0.1, 0.3, 0.5, 1.0]

**Outputs**:
- `results/comparison.csv`
- `results/selectivity_analysis.csv`
- Charts comparing all methods

**Dependencies**: Requires ALL other modules

---

## Configuration (Everyone uses this)

**File**: `config/config.yaml`

Key parameters:
- `data.embedding_method`: "OpenAI" (d=1536) or "MiniLM" (d=384)
- `index.nlist`: ~4√N (e.g., 1200 for N=100k)
- `search.nprobe_max`: 256
- `evaluation.k_values`: [10, 20]

---

## Git Workflow (Simple Version)

```bash
# 1. Create your branch
git checkout -b feature/your-module

# 2. Work on your files
# ... code ...

# 3. Commit frequently
git add .
git commit -m "feat(module): what you did"

# 4. Push your branch
git push origin feature/your-module

# 5. Create Pull Request on GitHub
# Target: develop branch
# Request review from: Zhangding Liu
```

**Commit message format**:
- `feat(data): add Wikipedia loader`
- `fix(index): correct signature logic`
- `test(search): add pruning tests`

---

## Testing

Each person should create tests in `tests/`:
- `tests/test_data.py` - Test data loading and metadata
- `tests/test_indexing.py` - Test index and signatures
- `tests/test_search.py` - Test pruning and search
- `tests/test_baselines.py` - Test baselines and evaluation

Run tests: `pytest tests/`

---

## Timeline

| Week | What | Who |
|------|------|-----|
| 1-2 | Data module | Yao-Ting |
| 1-3 | Indexing module | Zaowei |
| 2-3 | Search module | Yichang |
| 2-4 | Baselines & Evaluation | Zhangding |
| 4 | Integration & Experiments | Everyone |

---

## Need Help?
1. **API questions**: Check `docs/API_CONTRACT.md` (detailed specs)


