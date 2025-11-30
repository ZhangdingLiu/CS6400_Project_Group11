"""
IVF-PQ Index Wrapper

Owner: Zaowei Dai
Functionality: FAISS IVF-PQ index wrapper
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import faiss  # type: ignore


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
        if d <= 0 or nlist <= 0 or m <= 0 or nbits <= 0:
            raise ValueError("d, nlist, m, nbits must be positive.")
        if d % m != 0:
            raise ValueError("d must be divisible by m.")

        self.d = int(d)
        self.nlist = int(nlist)
        self.m = int(m)
        self.nbits = int(nbits)

        desc = f"IVF{self.nlist},PQ{self.m}x{self.nbits}"
        self.index: faiss.IndexIVF = faiss.index_factory(
            self.d, desc, faiss.METRIC_L2
        )  # type: ignore
        self.quantizer = self.index.quantizer
        self.ntotal: int = 0

    @staticmethod
    def _unit_float32(x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def _ensure_trained(self):
        if not self.index.is_trained:
            raise RuntimeError("Index is not trained.")

    def train(self, embeddings: np.ndarray):
        """
        Train the index.

        Args:
            embeddings: Shape (N, d)
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"embeddings must be (N, {self.d}).")
        X = self._unit_float32(embeddings)
        self.index.train(X)

    def add(self, embeddings: np.ndarray):
        """
        Add vectors to the index.

        Args:
            embeddings: Shape (N, d)
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"embeddings must be (N, {self.d}).")
        X = self._unit_float32(embeddings)
        self._ensure_trained()
        start = self.ntotal
        ids = np.arange(start, start + X.shape[0], dtype=np.int64)
        self.index.add_with_ids(X, ids)
        self.ntotal += X.shape[0]

    def get_list_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get IVF list assignment for each vector.

        Args:
            embeddings: Shape (N, d)

        Returns:
            np.ndarray: Shape (N,), dtype=int32, IVF list IDs
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"embeddings must be (N, {self.d}).")
        X = self._unit_float32(embeddings)
        self._ensure_trained()
        _, I = self.quantizer.assign(X)
        return I.astype(np.int32, copy=False)

    def get_centroids(self) -> np.ndarray:
        """
        Get IVF centroids.

        Returns:
            np.ndarray: Shape (nlist, d), centroid vectors
        """
        self._ensure_trained()
        xb = self.quantizer.xb
        arr = faiss.vector_to_array(xb)
        C = arr.reshape(self.nlist, self.d).astype(np.float32, copy=False)
        return C

    def search_preassigned(
        self, query: np.ndarray, list_ids: List[int], k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search only specified IVF lists.

        Args:
            query: Shape (d,), query vector
            list_ids: IVF list IDs to search
            k: Number of results to return

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, ids)
        """
        if query.ndim != 1 or query.shape[0] != self.d:
            raise ValueError(f"query must be ({self.d},).")
        if k <= 0:
            raise ValueError("k must be positive.")
        self._ensure_trained()

        q = query.astype(np.float32, copy=False)
        q /= (np.linalg.norm(q) + 1e-12)
        xq = q.reshape(1, -1)

        Dall, Iall = self.quantizer.search(xq, self.nlist)
        Dmap = {int(Iall[0, j]): float(Dall[0, j]) for j in range(self.nlist)}

        lids = np.asarray(list_ids, dtype=np.int64).reshape(1, -1)
        ldis = np.empty_like(lids, dtype=np.float32)
        C = None
        for j, lid in enumerate(list_ids):
            if int(lid) in Dmap:
                ldis[0, j] = Dmap[int(lid)]
            else:
                if C is None:
                    C = self.get_centroids()
                c = C[int(lid)]
                ldis[0, j] = np.float32(np.sum((c - q) ** 2))

        params = faiss.SearchParametersIVF()
        params.nprobe = len(list_ids)
        distances, ids = self.index.search_preassigned(
            xq, k, lids, ldis, params  # type: ignore
        )
        return distances[0], ids[0]

    def save(self, path: str):
        """
        Save index to file.

        Args:
            path: File path
        """
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """
        Load index from file.

        Args:
            path: File path
        """
        self.index = faiss.read_index(path)  # type: ignore
        self.quantizer = self.index.quantizer
        self.d = self.index.d
        self.nlist = self.index.nlist
        self.ntotal = self.index.ntotal
