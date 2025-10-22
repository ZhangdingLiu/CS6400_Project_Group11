# CS6400 Hybrid Vector Search (Group 11)

Efficient vector search combining semantic similarity with metadata filtering.

## Team Members

- **Zhangding Liu** - Baselines & Evaluation
- **Yao-Ting Huang** - Data Module
- **Zaowei Dai** - Indexing Module
- **Yichang Xu** - Search Module


## 📋 Important Documents - READ FIRST!

**All team members MUST read these before starting:**

1. **[docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)** - Git协作规范（分支策略、PR流程、commit规范）
2. **[docs/API_CONTRACT.md](docs/API_CONTRACT.md)** - 模块接口规范（确保代码能正确对接）
3. **[TASK_ASSIGNMENT.md](TASK_ASSIGNMENT.md)** - 个人任务分配

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
│   ├── GIT_WORKFLOW.md    # ⭐ Git协作规范 (必读!)
│   └── API_CONTRACT.md    # ⭐ 接口规范 (必读!)
├── data/              # Data loading & preprocessing (Yao-Ting)
├── indexing/          # IVF-PQ index & signatures (Zaowei)
├── search/            # Search engine (Yichang)
├── baselines/         # Baseline methods (Zhangding)
├── evaluation/        # Evaluation framework (Zhangding)
├── experiments/       # Experiment runners
├── utils/             # Utilities (shared)
├── config/            # Configuration files
├── TASK_ASSIGNMENT.md # ⭐ 任务分配 (必读!)
└── requirements.txt   # Python dependencies
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

**完整流程请参考 [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)**

简要步骤：
1. 阅读 `TASK_ASSIGNMENT.md` 了解你的任务
2. 阅读 `docs/API_CONTRACT.md` 了解接口规范
3. 从develop创建feature分支
4. 在你负责的模块文件夹中编写代码
5. 添加单元测试
6. 提交PR到develop分支（不是main！）
7. 等待code review后merge

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

