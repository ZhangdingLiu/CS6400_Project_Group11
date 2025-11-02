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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple vectors.

    Args:
        query: Shape (d,), query vector
        vectors: Shape (N, d), database vectors

    Returns:
        np.ndarray: Shape (N,), similarity scores
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    return np.dot(vectors_norm, query_norm)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        a: Vector 1
        b: Vector 2

    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(a - b)
