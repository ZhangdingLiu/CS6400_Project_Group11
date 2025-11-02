"""Test data generation helpers for development"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def generate_test_embeddings(n: int = 1000, d: int = 128) -> np.ndarray:
    """Generate random normalized embeddings"""
    embeddings = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def generate_test_metadata(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic metadata matching project schema"""
    return pd.DataFrame({
        'category': np.random.randint(1, 31, n, dtype=np.int32),
        'importance': np.random.randint(1, 101, n, dtype=np.int32),
        'year': np.random.randint(0, 101, n, dtype=np.int32),
        'paragraph_len': np.random.randint(50, 500, n, dtype=np.int32),
        'region': np.random.choice(['NA', 'EU', 'APAC', 'LATAM', 'AFR'], n)
    })


def generate_test_queries(embeddings: np.ndarray, metadata_df: pd.DataFrame,
                         n_queries: int = 10) -> List[Dict]:
    """Generate simple test queries"""
    queries = []
    n_samples = len(embeddings)

    for i in range(n_queries):
        idx = np.random.randint(0, n_samples)

        if i % 3 == 0:
            filter_dict = {
                'category': {'op': 'IN', 'values': [1, 5, 10, 15]}
            }
        elif i % 3 == 1:
            filter_dict = {
                'year': {'op': 'RANGE', 'min': 20, 'max': 80}
            }
        else:
            filter_dict = {
                'category': {'op': 'IN', 'values': [1, 2, 3]},
                'year': {'op': 'RANGE', 'min': 30, 'max': 70}
            }

        queries.append({
            'vector': embeddings[idx].copy(),
            'filter': filter_dict,
            'k': 10
        })

    return queries


class MockIVFPQIndex:
    """Minimal mock of IVFPQIndex for testing PostFilterANN"""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.d = embeddings.shape[1]
        self.n = embeddings.shape[0]

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simple brute-force search"""
        query = query.reshape(-1)
        similarities = np.dot(self.embeddings, query)

        k = min(k, self.n)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]

        return similarities[top_k_indices], top_k_indices
