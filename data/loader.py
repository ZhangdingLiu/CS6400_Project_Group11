"""
Wikipedia Dataset Loader

Owner: Yao-Ting Huang
Functionality: Load Wikipedia embedding dataset from HuggingFace
"""

from typing import Dict, Tuple, List, Optional
import json, gzip
import numpy as np
from huggingface_hub import hf_hub_download  # pip install huggingface_hub

# file name and embedding column for each source
class WikipediaDataLoader:
    """Download & parse Wikipedia embeddings; return raw (unnormalized) arrays."""

    def __init__(self, embedding_method: str = "OpenAI", text_field: str = "body"):
        self.HF = {
            "OpenAI":    ("wiki_openai.ndjson.gz", "text-embedding-ada-002"),  # 1536-d
            "MiniLM":    ("wiki_minilm.ndjson.gz", "all-MiniLM-L6-v2"),         # 384-d
            "GTE-small": ("wiki_gte.ndjson.gz",    "embedding"),                # 384-d
        }
        if embedding_method not in self.HF:
            raise ValueError(f"embedding_method must be one of {list(self.HF)}")
        self.embedding_method = embedding_method
        self.text_field = text_field

    def load(self, n_samples: Optional[int] = None) -> tuple[List[str], np.memmap]:
        """
        Returns:
            texts: list[str] length N'
            embeddings: np.memmap shape (N', d), dtype=float32 (NO normalization)
        """
        filename, emb_col = self.HF[self.embedding_method]
        path = hf_hub_download(
            repo_id="Supabase/wikipedia-en-embeddings",
            filename=filename,
            repo_type="dataset",
        )

        # Pre-allocate to the requested cap; we'll stop at N and optionally shrink.
        N_cap = 224_482 if n_samples is None else int(n_samples)
        d = 1536 if self.embedding_method == "OpenAI" else 384

        X = np.memmap("tmp_embeddings.dat", dtype="float32", mode="w+", shape=(N_cap, d))
        texts: List[str] = ["" for _ in range(N_cap)]
        text_keys = [self.text_field, "body", "text", "content"]

        i = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if i >= N_cap:                # <-- prevent out-of-bounds
                    break
                row = json.loads(line)
                e = row.get(emb_col)
                if e is None:
                    continue
                X[i, :] = np.asarray(e, dtype=np.float32)
                texts[i] = next((row[k] for k in text_keys if isinstance(row.get(k), str)), "")
                i += 1

        # If we filled fewer than N_cap, return a view on the filled prefix.
        # (memmap can't be resized; slicing returns a smaller memmap-like view.)
        X.flush()
        if i < N_cap:
            X_view = X[:i]
            texts = texts[:i]
        else:
            X_view = X

        return texts, X_view
