# CS6400 Hybrid Vector Search (Group 11)

Efficient vector search combining semantic similarity with metadata filtering.

## Team Members

- **Zhangding Liu** - Baselines & Evaluation
- **Yao-Ting Huang** - Data Module
- **Zaowei Dai** - Indexing Module
- **Yichang Xu** - Search Module


## 📋 Important Documents - READ FIRST!

**All team members MUST read these before starting:**

1. **[docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)** - Git collaboration workflow (branching strategy, PR process, commit conventions)
2. **[docs/API_CONTRACT.md](docs/API_CONTRACT.md)** - Module interface specifications (ensures code integration)
3. **[TASK_ASSIGNMENT.md](TASK_ASSIGNMENT.md)** - Individual task assignments

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ZhangdingLiu/CS6400_Project_Group11.git
cd CS6400_Project_Group11
pip install -r requirements.txt

# 2. Read the important docs above ⭐

# 3. Create your feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/your-module-name

# 4. Start coding following API_CONTRACT.md
```

## Project Structure

```
CS6400_Project_Group11/
├── docs/              # 📋 Documentation
│   ├── GIT_WORKFLOW.md    # ⭐ Git workflow (MUST READ!)
│   └── API_CONTRACT.md    # ⭐ Interface specs (MUST READ!)
├── data/              # Data loading & preprocessing (Yao-Ting)
├── indexing/          # IVF-PQ index & signatures (Zaowei)
├── search/            # Search engine (Yichang)
├── baselines/         # Baseline methods (Zhangding)
├── evaluation/        # Evaluation framework (Zhangding)
├── experiments/       # Experiment runners
├── utils/             # Utilities (shared)
├── config/            # Configuration files
├── TASK_ASSIGNMENT.md # ⭐ Task assignments (MUST READ!)
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

## Development Workflow

**Full process: See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)**

Quick steps:
1. Read `TASK_ASSIGNMENT.md` to understand your tasks
2. Read `docs/API_CONTRACT.md` for interface specifications
3. Create feature branch from develop
4. Write code in your assigned module folder
5. Add unit tests
6. Submit PR to develop branch (NOT main!)
7. Wait for code review and merge

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

