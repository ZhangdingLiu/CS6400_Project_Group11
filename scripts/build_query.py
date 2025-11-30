import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.query_generator import generate_queries

import json
import numpy as np
import pandas as pd


def main():
    # Paths where build_data wrote files
    emb_path = "data_files/embeddings.npy"
    meta_path = "data_files/metadata.parquet"

    print(f"Loading embeddings from: {emb_path}")
    embeddings = np.load(emb_path)   # (N, d)

    print(f"Loading metadata from: {meta_path}")
    metadata = pd.read_parquet(meta_path)

    # ----- Your desired query specification -----
    # Example: 3 selectivity bands, cycling across queries
    selectivity_ranges = [
        (0.01, 0.05),   # very selective
        (0.05, 0.15),   # medium
        (0.15, 0.40),   # high
    ]

    n_queries = 500  # Or 1000 if you prefer

    print(f"Generating {n_queries} queries ...")
    queries = generate_queries(
        embeddings=embeddings,
        metadata=metadata,
        n_queries=n_queries,
        selectivity_ranges=selectivity_ranges,
        k_value=10,
    )

    # Convert NumPy vectors → lists for JSON
    for q in queries:
        if hasattr(q["vector"], "tolist"):
            q["vector"] = q["vector"].tolist()

    out_path = "data_files/queries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(queries)} queries to {out_path}")

if __name__ == "__main__":
    main()