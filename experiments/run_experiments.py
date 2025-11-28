"""
Main Experiment Runner

Functionality: Run complete experiment pipeline
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.oracle import ExactSearchOracle
from evaluation.evaluator import HybridSearchEvaluator
from evaluation.metrics import compute_memory_usage
from baselines.prefilter_bruteforce import PreFilterBruteForce

# Try to import indexing modules (optional - may not be implemented yet)
try:
    from indexing.ivf_index import IVFPQIndex
    from baselines.postfilter_ann import PostFilterANN
    INDEXING_AVAILABLE = True
except ImportError:
    INDEXING_AVAILABLE = False
    print("Note: Indexing module not available. Skipping PostFilter baseline.")


def create_test_queries(embeddings: np.ndarray, metadata_df: pd.DataFrame, n_queries: int = 5):
    """
    Create simple test queries for evaluation

    Args:
        embeddings: Embedding vectors
        metadata_df: Metadata DataFrame
        n_queries: Number of test queries to create

    Returns:
        List of query dictionaries
    """
    queries = []
    np.random.seed(42)

    # Sample query indices
    query_indices = np.random.choice(len(embeddings), size=n_queries, replace=False)

    for idx in query_indices:
        # Use actual embedding as query vector
        query_vector = embeddings[idx].copy()

        # Create a simple filter
        # Example: category IN [1, 5], year RANGE [20, 80]
        filter_dict = {
            'category': {'op': 'IN', 'values': [1, 5]},
            'year': {'op': 'RANGE', 'min': 20, 'max': 80}
        }

        queries.append({
            'vector': query_vector,
            'filter': filter_dict
        })

    return queries


def main():
    """
    Run complete experiment pipeline.

    Steps:
    1. Load data
    2. Create test queries
    3. Initialize oracle and baselines
    4. Run evaluation
    5. Save results
    """
    print("="*60)
    print("CS6400 Hybrid Vector Search - Experiment Runner")
    print("="*60)

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    data_dir = "data_files"

    embeddings_path = os.path.join(data_dir, "embeddings.npy")
    metadata_path = os.path.join(data_dir, "metadata.parquet")

    if not os.path.exists(embeddings_path):
        print(f"ERROR: Embeddings not found at {embeddings_path}")
        print("Please run data generation first:")
        print("  python scripts/build_data.py")
        return

    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata not found at {metadata_path}")
        print("Please run data generation first:")
        print("  python scripts/build_data.py")
        return

    embeddings = np.load(embeddings_path)
    metadata = pd.read_parquet(metadata_path)

    print(f"  Loaded {len(embeddings)} embeddings, dimension={embeddings.shape[1]}")
    print(f"  Loaded {len(metadata)} metadata records")

    # Step 2: Create test queries
    print("\n[2/5] Creating test queries...")
    queries = create_test_queries(embeddings, metadata, n_queries=5)
    print(f"  Created {len(queries)} test queries")

    # Step 3: Build index (if available) and initialize methods
    print("\n[3/5] Building index and initializing methods...")

    index = None
    index_build_time = None

    if INDEXING_AVAILABLE:
        print("  Building IVF-PQ index...")
        index_dir = "index_files"
        os.makedirs(index_dir, exist_ok=True)

        # Get index parameters from config or use defaults
        d = embeddings.shape[1]
        nlist = int(4 * np.sqrt(len(embeddings)))  # Approximately 4*sqrt(N)

        # Build index with timing
        start_time = time.time()
        try:
            index = IVFPQIndex(d=d, nlist=nlist, m=64, nbits=8)
            index.train(embeddings)
            index.add(embeddings)
            index_build_time = time.time() - start_time

            print(f"  Index built successfully in {index_build_time:.2f} seconds")

            # Save index
            index_path = os.path.join(index_dir, "index.faiss")
            index.save(index_path)
            print(f"  Index saved to: {index_path}")

        except Exception as e:
            print(f"  Warning: Failed to build index: {e}")
            index = None
            index_build_time = None
    else:
        print("  Skipping index build (indexing module not available)")

    # Initialize evaluation methods
    oracle = ExactSearchOracle(embeddings, metadata)
    prefilter = PreFilterBruteForce(embeddings, metadata)

    methods_list = ['Oracle', 'PreFilter']

    # Add PostFilter if index is available
    postfilter = None
    if index is not None:
        try:
            postfilter = PostFilterANN(index, metadata, fixed_nprobe=32)
            methods_list.append('PostFilter')
        except Exception as e:
            print(f"  Warning: Failed to initialize PostFilter: {e}")

    print(f"  Initialized methods: {', '.join(methods_list)}")

    # Step 4: Run evaluation
    print("\n[4/5] Running evaluation...")
    evaluator = HybridSearchEvaluator(queries, oracle)

    methods = {
        'Oracle': oracle,
        'PreFilter': prefilter
    }

    if postfilter is not None:
        methods['PostFilter'] = postfilter

    results = evaluator.compare_methods(methods, k_values=[10, 20])

    # Step 5: Save results
    print("\n[5/5] Saving results...")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "comparison.csv")
    results.to_csv(output_path, index=False)

    print(f"  Results saved to: {output_path}")

    # Save index build time and memory usage if available
    if index_build_time is not None or index is not None:
        index_stats = {}

        if index_build_time is not None:
            index_stats['index_build_time_seconds'] = index_build_time

        # Compute memory usage
        try:
            memory_stats = compute_memory_usage("index_files")
            index_stats.update(memory_stats)
        except Exception as e:
            print(f"  Warning: Failed to compute memory usage: {e}")

        if index_stats:
            stats_path = os.path.join(results_dir, "index_stats.csv")
            pd.DataFrame([index_stats]).to_csv(stats_path, index=False)
            print(f"  Index statistics saved to: {stats_path}")

    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results.to_string(index=False))

    if index_build_time is not None:
        print("\n" + "="*60)
        print("INDEX BUILD TIME")
        print("="*60)
        print(f"  Build time: {index_build_time:.2f} seconds")

        # Display memory usage if available
        try:
            mem_stats = compute_memory_usage("index_files")
            print(f"  Index size: {mem_stats['index_size_mb']:.2f} MB")
            print(f"  Signature size: {mem_stats['signature_size_mb']:.2f} MB")
            print(f"  Total size: {mem_stats['total_size_mb']:.2f} MB")
        except:
            pass

    print("\n" + "="*60)
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
