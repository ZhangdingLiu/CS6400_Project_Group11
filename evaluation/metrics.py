"""
Evaluation Metrics

Owner: Zhangding Liu
Functionality: Compute evaluation metrics including recall and latency
"""

from typing import List, Dict
import numpy as np


def recall_at_k(retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray) -> float:
    """
    Compute Recall@k.

    Args:
        retrieved_ids: Shape (k,), retrieved IDs
        ground_truth_ids: Shape (k,), ground truth IDs

    Returns:
        float: Recall value [0.0, 1.0]
    """
    raise NotImplementedError


def compute_latency_stats(latencies: List[float]) -> Dict:
    """
    Compute latency statistics.

    Args:
        latencies: List of latency values (milliseconds)

    Returns:
        Dict: Statistics including p50, p95, p99, mean
    """
    raise NotImplementedError
