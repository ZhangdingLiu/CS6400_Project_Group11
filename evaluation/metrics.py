"""
Evaluation Metrics

Owner: Zhangding Liu
Functionality: Compute evaluation metrics including recall and latency
"""

from typing import List, Dict, Optional
import numpy as np
import os


def recall_at_k(retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray) -> float:
    """
    Compute Recall@k.

    Args:
        retrieved_ids: Shape (k,), retrieved IDs
        ground_truth_ids: Shape (k,), ground truth IDs

    Returns:
        float: Recall value [0.0, 1.0]
    """
    if len(ground_truth_ids) == 0:
        return 0.0

    intersection = set(retrieved_ids) & set(ground_truth_ids)
    return len(intersection) / len(ground_truth_ids)


def compute_latency_stats(latencies: List[float]) -> Dict:
    """
    Compute latency statistics.

    Args:
        latencies: List of latency values (milliseconds)

    Returns:
        Dict: Statistics including p50, p95, p99, mean
    """
    if not latencies:
        return {
            'mean': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0,
            'min': 0.0,
            'max': 0.0
        }

    latencies_array = np.array(latencies)
    return {
        'mean': float(np.mean(latencies_array)),
        'p50': float(np.percentile(latencies_array, 50)),
        'p95': float(np.percentile(latencies_array, 95)),
        'p99': float(np.percentile(latencies_array, 99)),
        'min': float(np.min(latencies_array)),
        'max': float(np.max(latencies_array))
    }


def compute_memory_usage(index_dir: str = "index_files") -> Dict:
    """
    Compute memory usage of index and signatures.

    Args:
        index_dir: Directory containing index files

    Returns:
        Dict: Memory usage statistics
            - index_size_mb: Index file size in MB
            - signature_size_mb: Signature file size in MB
            - total_size_mb: Total size in MB
    """
    def get_file_size_mb(filepath: str) -> float:
        """Get file size in MB"""
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / (1024 * 1024)
        return 0.0

    index_size = 0.0
    signature_size = 0.0

    # Check for FAISS index file
    index_path = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_path):
        index_size = get_file_size_mb(index_path)

    # Check for signature file
    signature_path = os.path.join(index_dir, "signatures.parquet")
    if os.path.exists(signature_path):
        signature_size = get_file_size_mb(signature_path)

    return {
        'index_size_mb': float(index_size),
        'signature_size_mb': float(signature_size),
        'total_size_mb': float(index_size + signature_size)
    }
