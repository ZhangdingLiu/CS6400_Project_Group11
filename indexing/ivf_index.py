"""
IVF-PQ Index Wrapper

Owner: Zaowei Dai
Functionality: FAISS IVF-PQ index wrapper
"""

from typing import List, Tuple
import numpy as np


class IVFPQIndex:
    """Wrapper around FAISS IVF-PQ index"""

    def __init__(self, d: int, nlist: int, m: int = 64, nbits: int = 8):
        """
        Initialize IVF-PQ index.

        Args:
            d: Embedding dimension
            nlist: Number of IVF lists
            m: Number of PQ subquantizers
            nbits: Bits per subquantizer
        """
        self.d = d
        self.nlist = nlist
        self.m = m
        self.nbits = nbits

    def train(self, embeddings: np.ndarray):
        """
        Train the index.

        Args:
            embeddings: Shape (N, d), normalized embeddings
        """
        raise NotImplementedError

    def add(self, embeddings: np.ndarray):
        """
        Add vectors to the index.

        Args:
            embeddings: Shape (N, d), normalized embeddings
        """
        raise NotImplementedError

    def get_list_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get IVF list assignment for each vector.

        Args:
            embeddings: Shape (N, d)

        Returns:
            np.ndarray: Shape (N,), dtype=int32, IVF list IDs
        """
        raise NotImplementedError

    def get_centroids(self) -> np.ndarray:
        """
        Get IVF centroids.

        Returns:
            np.ndarray: Shape (nlist, d), centroid vectors
        """
        raise NotImplementedError

    def search_preassigned(self, query: np.ndarray, list_ids: List[int], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search only specified IVF lists.

        Args:
            query: Shape (d,), query vector
            list_ids: IVF list IDs to search
            k: Number of results to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, ids)
        """
        raise NotImplementedError

    def save(self, path: str):
        """
        Save index to file.

        Args:
            path: File path
        """
        raise NotImplementedError

    def load(self, path: str):
        """
        Load index from file.

        Args:
            path: File path
        """
        raise NotImplementedError
