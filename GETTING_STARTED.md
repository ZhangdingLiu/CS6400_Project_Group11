# Getting Started - For Team Members

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd CS6400_Project_Group11
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Read Your Task

Open `TASK_ASSIGNMENT.md` and find your section:
- **Yao-Ting**: Task 1 - Data Module
- **Zaowei**: Task 2 - Indexing Module (Critical!)
- **Yichang**: Task 3 - Search Module
- **Zhangding**: Task 4 - Baselines & Evaluation

## Step 4: Create Your Branch

```bash
git checkout -b feature/your-module-name
```

Examples:
- `feature/data-loader`
- `feature/ivf-index`
- `feature/search-engine`
- `feature/baselines`

## Step 5: Create Your Files

Work in your assigned folder and create the required Python files listed in `TASK_ASSIGNMENT.md`.

## Step 6: Test Your Code

```bash
# Test as you develop
python your_file.py

# Or create tests (recommended)
pytest
```

## Step 7: Commit Your Changes

```bash
git add .
git commit -m "feat(module): brief description"
git push origin feature/your-module-name
```

## Step 8: Create Pull Request

1. Go to GitHub repository
2. Click "Pull Requests"
3. Click "New Pull Request"
4. Select your branch
5. Add description of what you implemented
6. Request review from Zhangding Liu

## Need Help?

- **Can't find something?**: Check `TASK_ASSIGNMENT.md`
- **Stuck on implementation?**: Ask in team chat
- **Git problems?**: Google or ask teammates
- **Integration issues?**: Discuss with affected team member

## That's It!

Simple workflow:
1. Read task
2. Create branch
3. Write code
4. Test
5. Commit
6. Pull request

