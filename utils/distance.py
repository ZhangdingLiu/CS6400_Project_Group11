"""
Distance Computation Utilities

Functionality: Distance and similarity computation
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: Vector 1
        b: Vector 2

    Returns:
        float: Cosine similarity
    """
    raise NotImplementedError


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple vectors.

    Args:
        query: Shape (d,), query vector
        vectors: Shape (N, d), database vectors

    Returns:
        np.ndarray: Shape (N,), similarity scores
    """
    raise NotImplementedError


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        a: Vector 1
        b: Vector 2

    Returns:
        float: Euclidean distance
    """
    raise NotImplementedError
