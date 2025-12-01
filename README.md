# CS6400 Hybrid Vector Search (Group 11)

Efficient vector search combining semantic similarity with metadata filtering.

## Team Members

- **Zhangding Liu** - Baselines & Evaluation
- **Yao-Ting Huang** - Data Module
- **Zaowei Dai** - Indexing Module
- **Yichang Xu** - Search Module

## Code-to-Report Mapping

This section maps the main components described in our report to their code implementations. All files are NEW (written from scratch for this project).

### Data Generation & Preprocessing
- `data/loader.py`: Wikipedia embedding dataset loader (OpenAI ada-002, d=1536)
- `data/preprocessor.py`: Synthetic metadata generation (category, importance, year, region)
- `data/query_generator.py`: Query workload generation with controllable selectivity
- `scripts/build_data.py`: Dataset generation script (100k samples default)
- `scripts/build_query.py`: Query generation script (500 queries)

### IVF-PQ Indexing with Metadata Signatures
- `indexing/ivf_index.py`: FAISS IVF-PQ index wrapper with selective list search
- `indexing/metadata_signatures.py`: Per-list metadata signature builder
  - Categorical signatures (bitsets)
  - Numeric signatures (min/max ranges + bucket masks)
- `indexing/signature_storage.py`: Signature persistence using Parquet

### Filter-Aware Hybrid Search
- `search/pruning.py`: Filter-aware IVF list pruning using metadata signatures
- `search/adaptive_deepening.py`: Adaptive nprobe and k' parameter adjustment
- `search/search_engine.py`: Main hybrid search coordinator

### Baseline Implementations
- `baselines/prefilter_bruteforce.py`: Pre-filter baseline (filter → exact search)
- `baselines/postfilter_ann.py`: Post-filter baseline (ANN → filter → retry logic)

### Evaluation & Experiments
- `evaluation/oracle.py`: Exact filtered search for ground truth
- `evaluation/metrics.py`: Recall@k, latency (P50/P95/P99), memory usage
- `evaluation/evaluator.py`: Experiment framework with selectivity binning
- `experiments/run_experiments.py`: Main experiment runner (k=[10,20,50])
- `experiments/analyze_results.py`: Result visualization (7 individual plots for LaTeX)

### Generated Results
- `results/comparison.csv`: Method comparison across k=[10,20,50]
- `results/selectivity_analysis.csv`: Performance by selectivity bins
- `results/index_stats.csv`: Index build time and memory overhead
- `results/figures/exp1_*.png`: Method comparison plots (4 files)
- `results/figures/exp2_*.png`: Selectivity analysis plots (3 files)

## Quick Start

```bash
# Clone and install
git clone https://github.com/ZhangdingLiu/CS6400_Project_Group11.git
cd CS6400_Project_Group11
pip install -r requirements.txt

# Run experiments (see "Running Experiments" section below)
python scripts/build_data.py
python scripts/build_query.py
python experiments/run_experiments.py
python experiments/analyze_results.py
```

## Project Structure

```
CS6400_Project_Group11/
├── scripts/           # Data generation scripts
├── data/              # Data loading & preprocessing
├── indexing/          # IVF-PQ index & metadata signatures
├── search/            # Filter-aware pruning & adaptive deepening
├── baselines/         # PreFilter, PostFilter baselines
├── evaluation/        # Evaluation metrics & framework
├── experiments/       # Experiment runners & result analysis
├── results/           # Experiment results & figures
└── requirements.txt   # Python dependencies
```

## How It Works

1. **Filter-Aware Pruning**: Metadata signatures eliminate IVF lists that can't satisfy filters
2. **Adaptive Deepening**: Dynamically adjust search parameters based on intermediate results
3. **Hybrid Search**: Combines vector similarity with structured metadata filtering

## Running Experiments

### Quick Start (100k dataset)
```bash
# Step 1: Generate dataset (default: 100k samples, ~586MB)
python scripts/build_data.py

# Step 2: Generate queries (500 queries)
python scripts/build_query.py

# Step 3: Run experiments (~8-15 min)
python experiments/run_experiments.py

# Step 4: Generate visualizations
python experiments/analyze_results.py
```

### Custom Dataset Size
```bash
# 10k samples (quick test)
python scripts/build_data.py --n-samples 10000

# 50k samples
python scripts/build_data.py --n-samples 50000

# Then run steps 2-4 as above
```

### Output Files
- **Data**: `data_files/embeddings.npy`, `metadata.parquet`, `queries.json`
- **Results**: `results/comparison.csv`, `selectivity_analysis.csv`, `index_stats.csv`
- **Figures**: `results/figures/exp1_*.png`, `exp2_*.png` (7 individual plots for LaTeX)

## Evaluation Metrics

- **Recall@k** (k=10, 20, 50): Search quality
- **Latency** (P50/P95/Mean): Query performance
- **Index Build Time**: Indexing efficiency
- **Memory Usage**: Index + signature overhead

## Key Technologies

- **FAISS**: IVF-PQ indexing for efficient vector search
- **DuckDB**: Metadata filtering
- **NumPy/Pandas**: Data processing
- **Matplotlib**: Result visualization

See `requirements.txt` for full dependencies.

