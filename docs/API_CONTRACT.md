# API Contract

本文档定义各模块之间的接口规范，确保团队成员实现的模块能够正确对接。

## 数据格式规范

### Filter Dictionary
所有模块统一使用以下格式表示过滤条件：
```python
filter_dict = {
    'category': {'op': 'IN', 'values': [1, 5, 10]},
    'year': {'op': 'RANGE', 'min': 20, 'max': 80},
    'importance': {'op': 'GT', 'value': 50},
    'region': {'op': 'EQ', 'value': 'EU'}
}
```

支持的操作符：
- `IN`: 值在列表中
- `RANGE`: 范围查询 (min, max)
- `GT/LT/GTE/LTE`: 大于/小于/大于等于/小于等于
- `EQ`: 等于

### Metadata DataFrame
5列固定格式 (与embeddings按行对应)：
- `category`: int32, [1, 30]
- `importance`: int32, [1, 100]
- `year`: int32, [0, 100]
- `paragraph_len`: int32
- `region`: str, {NA, EU, APAC, LATAM, AFR}

---

## 模块接口

### 1. Data Module (Yao-Ting Huang)

#### `data/loader.py`
```python
def load_embeddings(dataset_name: str, n_samples: int) -> tuple:
    """
    从HuggingFace加载数据

    Returns:
        embeddings: np.ndarray (N, d), float32, L2-normalized
        texts: list[str], 长度N
    """
```

#### `data/preprocessor.py`
```python
def generate_metadata(n_samples: int, config: dict) -> pd.DataFrame:
    """
    生成合成metadata

    Returns:
        DataFrame with 5 columns (category, importance, year, paragraph_len, region)
    """
```

#### `data/query_generator.py`
```python
def generate_queries(embeddings, metadata, n_queries: int,
                     selectivity_ranges: list) -> list[dict]:
    """
    生成查询workload

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
        """初始化IVF-PQ索引"""

    def train(self, embeddings: np.ndarray):
        """训练索引"""

    def add(self, embeddings: np.ndarray):
        """添加向量到索引"""

    def get_list_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        获取每个向量所属的IVF list ID

        Returns:
            np.ndarray (N,), dtype=int, 值范围 [0, nlist-1]
        """

    def get_centroids(self) -> np.ndarray:
        """
        Returns:
            np.ndarray (nlist, d), centroid vectors
        """

    def search_preassigned(self, query: np.ndarray, list_ids: list[int],
                          k_prime: int) -> tuple:
        """
        只搜索指定的IVF lists

        Args:
            query: (d,) query vector
            list_ids: 要搜索的list IDs
            k_prime: 返回top-k'个结果

        Returns:
            distances: np.ndarray (k_prime,)
            ids: np.ndarray (k_prime,), 向量在数据集中的索引
        """

    def save(self, path: str):
        """保存索引"""

    def load(self, path: str):
        """加载索引"""
```

#### `indexing/metadata_signatures.py`
```python
class MetadataSignatureBuilder:
    def build_signatures(self, metadata: pd.DataFrame,
                        list_assignments: np.ndarray) -> dict:
        """
        为每个IVF list构建metadata签名

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
        """初始化pruner"""

    def apply_filter(self, filter_dict: dict) -> list[int]:
        """
        根据filter pruning IVF lists

        Returns:
            candidate_list_ids: 可能包含结果的list IDs
        """

    def rank_lists_by_distance(self, query: np.ndarray,
                              candidate_lists: list[int]) -> list[int]:
        """
        按query到centroid距离排序lists

        Returns:
            sorted_list_ids: 按距离从近到远排序
        """
```

#### `search/adaptive_deepening.py`
```python
class AdaptiveSearchPlanner:
    def initialize_parameters(self, k: int,
                            estimated_selectivity: float) -> tuple[int, int]:
        """
        初始化nprobe和k_prime

        Returns:
            nprobe_0, k_prime_0
        """

    def should_deepen(self, n_results: int, k: int) -> bool:
        """判断是否需要继续deepening"""

    def grow_parameters(self, nprobe: int, k_prime: int) -> tuple[int, int]:
        """
        增长参数

        Returns:
            new_nprobe, new_k_prime
        """
```

#### `search/search_engine.py`
```python
class HybridSearchEngine:
    def __init__(self, index: IVFPQIndex, metadata: pd.DataFrame,
                 signatures: dict, config: dict):
        """初始化搜索引擎"""

    def search(self, query: np.ndarray, filter_dict: dict,
              k: int) -> tuple:
        """
        混合搜索主接口

        Args:
            query: (d,) normalized query vector
            filter_dict: metadata过滤条件
            k: 返回top-k个结果

        Returns:
            distances: np.ndarray (<=k,)
            ids: np.ndarray (<=k,), 满足filter的结果ID
            stats: dict, 包含 {lists_probed, codes_scored, iterations}
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
        Pre-filter baseline: 先filter再精确搜索

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
        Post-filter baseline: ANN搜索后filter，不足则retry

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
        精确搜索ground truth

        Returns:
            distances, ids, stats
        """
```

#### `evaluation/metrics.py`
```python
def compute_recall(predicted_ids: np.ndarray,
                   ground_truth_ids: np.ndarray, k: int) -> float:
    """计算recall@k"""

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
        对比多个方法

        Returns:
            DataFrame with columns: [method, selectivity_range, recall@k, p50, p95, p99]
        """
```

---

## 集成检查清单

在合并代码前，请确保：

- [ ] 函数签名与API_CONTRACT完全一致
- [ ] 返回值类型和格式符合规范
- [ ] 添加了对应的单元测试
- [ ] filter_dict格式统一
- [ ] metadata DataFrame列顺序正确
- [ ] 向量已L2归一化（使用cosine similarity）
