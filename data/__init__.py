"""Data loading and preprocessing module."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from numpy.lib.format import open_memmap  # for writing .npy without huge RAM use

# Local imports (relative)
from .loader import WikipediaDataLoader
from .preprocessor import MetadataGenerator

__all__ = ["WikipediaDataLoader", "MetadataGenerator", "build_data_files"]
__version__ = "0.1.0"

def build_data_files(
    loader : WikipediaDataLoader,
    out_dir: str = "data_files",
    n_samples: Optional[int] = None,
  ) -> tuple[str, str]:
    """
    Creates:
        - {out_dir}/embeddings.npy     (L2-normalized float32)
        - {out_dir}/metadata.parquet   (5 columns)
    Returns:
        (embeddings_path, metadata_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    texts, X = loader.load(n_samples=n_samples)  # X is memmap (N,d), unnormalized
    N, d = X.shape

    # 1) Normalize row-wise into a .npy file without loading all into RAM
    emb_path = os.path.join(out_dir, "embeddings.npy")
    # create an on-disk .npy we can write into
    Y = open_memmap(emb_path, mode="w+", dtype="float32", shape=(N, d))
    np.copyto(Y, X, casting="no")

    # ensure written
    del Y  # closes memmap
    X._mmap.close()  # close tmp memmap file
    try:
        os.remove("tmp_embeddings.dat")
    except OSError:
        pass

    # 2) Build & save metadata parquet
    meta = MetadataGenerator(seed=42).build_metadata_table(N, texts)
    meta_path = os.path.join(out_dir, "metadata.parquet")
    # requires pyarrow or fastparquet installed via requirements.txt
    meta.to_parquet(meta_path, index=False)

    return emb_path, meta_path
