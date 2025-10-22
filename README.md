# CS6400 Hybrid Vector Search (Group 11)

Efficient vector search combining semantic similarity with metadata filtering.

## Team Members

- **Yao-Ting Huang** - Data Module
- **Zaowei Dai** - Indexing Module
- **Yichang Xu** - Search Module
- **Zhangding Liu** - Baselines & Evaluation

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd CS6400_Project_Group11
pip install -r requirements.txt

# 2. Read your task
See TASK_ASSIGNMENT.md

# 3. Create your branch and start coding
git checkout -b feature/your-module
```

## Project Structure

```
CS6400_Project_Group11/
├── data/              # Data loading & preprocessing (Yao-Ting)
├── indexing/          # IVF-PQ index & signatures (Zaowei)
├── search/            # Search engine (Yichang)
├── baselines/         # Baseline methods (Zhangding)
├── evaluation/        # Evaluation framework (Zhangding)
├── experiments/       # Experiment runners (Zhangding)
├── utils/             # Utilities (shared)
├── config/            # Configuration files
└── TASK_ASSIGNMENT.md # ⭐ READ THIS FIRST
```

## How It Works

1. **Filter-Aware Pruning**: Metadata signatures eliminate IVF lists that can't satisfy filters
2. **Adaptive Deepening**: Dynamically adjust search parameters based on intermediate results
3. **Hybrid Search**: Combines vector similarity with structured metadata filtering

## Running Experiments

```bash
# After all modules are implemented:
python experiments/run_experiments.py
python experiments/analyze_results.py
```

## Development Workflow

1. Read `TASK_ASSIGNMENT.md` for your specific task
2. Create files in your assigned folder
3. Implement required functions
4. Test your module
5. Submit Pull Request

## Configuration

Edit `config/config.yaml` to adjust:
- Dataset size and embedding method
- Index parameters (nlist, m, nbits)
- Search parameters (nprobe_max, growth factors)

## Dependencies

- FAISS (vector indexing)
- NumPy, Pandas (data processing)
- PyArrow (Parquet files)
- PyTest (testing)

See `requirements.txt` for full list.

## Timeline

- **Week 1-2**: Data + Indexing
- **Week 2-3**: Search + Baselines
- **Week 4**: Integration + Experiments

## Questions?

