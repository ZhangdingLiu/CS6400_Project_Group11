import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.oracle import ExactSearchOracle
from evaluation.evaluator import HybridSearchEvaluator
from evaluation.metrics import compute_memory_usage
from baselines.prefilter_bruteforce import PreFilterBruteForce
from baselines.postfilter_ann import PostFilterANN
from indexing.ivf_index import IVFPQIndex
from indexing.metadata_signatures import MetadataSignatureBuilder
from search.pruning import FilterAwarePruner
from search.adaptive_deepening import AdaptiveSearchPlanner
from search.search_engine import HybridSearchEngine


def load_queries(query_path):
    with open(query_path, 'r') as f:
        queries = json.load(f)
    for q in queries:
        q['vector'] = np.array(q['vector'], dtype=np.float32)
    return queries


def build_index_and_signatures(embeddings, metadata):
    d = embeddings.shape[1]
    nlist = int(4 * np.sqrt(len(embeddings)))

    print(f"Building IVF-PQ index (nlist={nlist})...")
    start = time.time()
    index = IVFPQIndex(d=d, nlist=nlist, m=64, nbits=8)
    index.train(embeddings)
    index.add(embeddings)
    build_time = time.time() - start

    print("Building metadata signatures...")
    assignments = index.get_list_assignments(embeddings)
    builder = MetadataSignatureBuilder(metadata, assignments)
    raw_sigs = builder.build_all_signatures()

    # Transform signature format for FilterAwarePruner
    signatures = {}
    if 'numeric' in raw_sigs:
        signatures.update(raw_sigs['numeric'])
    if 'categorical' in raw_sigs:
        signatures.update(raw_sigs['categorical'])

    return index, signatures, build_time


def main():
    data_dir = "data_files"
    index_dir = "index_files"
    results_dir = "results"
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("Hybrid Vector Search Experiments")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...")
    embeddings = np.load(f"{data_dir}/embeddings.npy")
    metadata = pd.read_parquet(f"{data_dir}/metadata.parquet")
    queries = load_queries(f"{data_dir}/queries.json")
    print(f"  Data: {len(embeddings)} vectors, {len(queries)} queries")

    # Build index
    print("\n[2/4] Building index and signatures...")
    index, signatures, build_time = build_index_and_signatures(embeddings, metadata)
    index.save(f"{index_dir}/index.faiss")
    print(f"  Build time: {build_time:.2f}s")

    # Initialize methods
    print("\n[3/4] Initializing methods...")
    oracle = ExactSearchOracle(embeddings, metadata)
    prefilter = PreFilterBruteForce(embeddings, metadata)
    postfilter = PostFilterANN(index, metadata, fixed_nprobe=32)

    centroids = index.get_centroids()
    pruner = FilterAwarePruner(signatures, centroids)
    planner = AdaptiveSearchPlanner(
        nlist=index.nlist,
        nprobe_max=256,
        growth_factor_nprobe=1.8,
        growth_factor_k_prime=1.5
    )
    hybridsearch = HybridSearchEngine(index, pruner, planner, metadata)

    methods = {
        'Oracle': oracle,
        'PreFilter': prefilter,
        'PostFilter': postfilter,
        'HybridSearch': hybridsearch
    }

    # Run evaluation
    print("\n[4/4] Running experiments...")
    evaluator = HybridSearchEvaluator(queries, oracle)

    # Experiment 1: Method comparison
    print("  - Method comparison...")
    results = evaluator.compare_methods(methods, k_values=[10, 20, 50])
    results.to_csv(f"{results_dir}/comparison.csv", index=False)

    # Experiment 2: Selectivity analysis
    print("  - Selectivity analysis...")
    selectivity_bins = [(0.01, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.00)]
    sel_results = []

    for low, high in selectivity_bins:
        bin_queries = [q for q in queries if low <= q.get('selectivity', 0) < high]
        if not bin_queries:
            continue

        bin_eval = HybridSearchEvaluator(bin_queries, oracle)
        bin_res = bin_eval.compare_methods(methods, k_values=[10, 20, 50])
        bin_res['selectivity_bin'] = f"{low}-{high}"
        bin_res['n_queries'] = len(bin_queries)
        sel_results.append(bin_res)

    if sel_results:
        pd.concat(sel_results, ignore_index=True).to_csv(
            f"{results_dir}/selectivity_analysis.csv", index=False
        )

    # Experiment 3: Memory overhead
    mem_stats = compute_memory_usage(index_dir)
    mem_stats['index_build_time_seconds'] = build_time
    pd.DataFrame([mem_stats]).to_csv(f"{results_dir}/index_stats.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    print(f"\nIndex build time: {build_time:.2f}s")
    print(f"Index size: {mem_stats['index_size_mb']:.2f} MB")
    print(f"Signature size: {mem_stats['signature_size_mb']:.2f} MB")
    print("\n" + "="*60)
    print("Experiments complete!")


if __name__ == '__main__':
    main()
