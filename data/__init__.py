"""Data loading and preprocessing module."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from numpy.lib.format import open_memmap  # for writing .npy without huge RAM use

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
        - {out_dir}/embeddings.npy     
        - {out_dir}/metadata.parquet   
    Returns:
        (embeddings_path, metadata_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    texts, X = loader.load(n_samples=n_samples)  
    N, d = X.shape

    emb_path = os.path.join(out_dir, "embeddings.npy")
    # create an on-disk .npy we can write into
    Y = open_memmap(emb_path, mode="w+", dtype="float32", shape=(N, d))
    np.copyto(Y, X, casting="no")

    # ensure written
    del Y 
    X._mmap.close()  
    try:
        os.remove("tmp_embeddings.dat")
    except OSError:
        pass

    # Build and save metadata parquet
    meta = MetadataGenerator(seed=42).build_metadata_table(N, texts)
    meta_path = os.path.join(out_dir, "metadata.parquet")
    # requires pyarrow or fastparquet installed 
    meta.to_parquet(meta_path, index=False)

    return emb_path, meta_path
