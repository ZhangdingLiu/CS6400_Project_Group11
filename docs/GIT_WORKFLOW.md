# Git Workflow Documentation

This document defines the Git collaboration standards for CS6400 Group 11 project. All team members must follow these guidelines.

## Branching Strategy

### Branch Types

```
main (production branch)
  └── develop (development integration branch)
       ├── feature/data-loader (feature branch - Yao-Ting)
       ├── feature/indexing-ivf (feature branch - Zaowei)
       ├── feature/search-pruning (feature branch - Yichang)
       └── feature/baselines-eval (feature branch - Zhangding)
```

**Branch Descriptions:**
- **main**: Stable milestone version, only updated at important project milestones
- **develop**: Development integration branch, all feature branches merge here
- **feature/\<module\>-\<name\>**: Individual feature branches, created from develop

### Branch Rules

1. ✅ **Allowed Operations**:
   - Create feature branches from develop
   - Merge feature branches to develop
   - Merge develop to main at milestones

2. ❌ **Prohibited Operations**:
   - Direct push to main branch
   - Direct merge feature branches to main
   - Direct merge across feature branches

---

## Complete Workflow

### Step 1: Starting a New Task

```bash
# 1. Ensure you're on develop branch
git checkout develop

# 2. Pull latest code (avoid developing on outdated code)
git pull origin develop

# 3. Create feature branch
git checkout -b feature/<module>-<description>

# Examples:
git checkout -b feature/data-loader
git checkout -b feature/indexing-signatures
git checkout -b feature/search-pruning
git checkout -b feature/baseline-postfilter
```

### Step 2: During Development

```bash
# 1. Write code...

# 2. View changes
git status
git diff

# 3. Add changes to staging area
git add <filename>           # Add specific file
git add .                    # Add all changes (use with caution)

# 4. Commit (follow commit conventions)
git commit -m "feat(data): implement Zipf distribution for category generation"

# 5. Push to remote regularly (backup + let team see progress)
git push origin feature/<your-branch-name>
```

**Development Recommendations:**
- Commit frequently (commit after completing each small feature)
- Push at least once per day (avoid code loss)
- commit messages should clearly describe what was done

### Step 3: Completing Development, Preparing to Merge

```bash
# 1. Ensure all changes are committed
git status  # Should show "nothing to commit, working tree clean"

# 2. Pull latest develop (others may have merged code)
git checkout develop
git pull origin develop

# 3. Switch back to feature branch
git checkout feature/<your-branch-name>

# 4. Merge develop's latest code into your branch (resolve potential conflicts)
git merge develop

# 5. If there are conflicts, resolve them and commit
git add <resolved-file>
git commit -m "merge: resolve conflicts with develop"

# 6. Push to remote
git push origin feature/<your-branch-name>
```

### Step 4: Creating Pull Request (PR)

**On GitHub web interface:**

1. Visit: https://github.com/ZhangdingLiu/CS6400_Project_Group11
2. Click **"Pull requests"** → **"New pull request"**
3. Set branches:
   - **base**: `develop` ← merge target
   - **compare**: `feature/<your-branch-name>` ← your feature branch
4. Fill in PR title and description:

```markdown
## Completed Work
- [x] Implemented XXX feature
- [x] Added unit tests
- [x] Updated relevant documentation

## Testing Status
- [x] All unit tests pass (`pytest tests/test_xxx.py`)
- [x] Function signatures match API_CONTRACT.md
- [x] Code follows Python standards (type hints, docstrings)

## Interface Verification
- [x] Input/output format conforms to API_CONTRACT specs
- [x] Matches dependent module interfaces

## Notes
(Describe anything reviewers should pay attention to)
```

5. Request team member review (select at least 1 person in Reviewers on the right)
6. Wait for review feedback

### Step 5: Code Review

**As PR Creator:**
- Modify code based on review comments
- Commit and push on feature branch (PR will auto-update)
- Reply to reviewer's comments

**As Reviewer:**
- Check code quality, test coverage, interface consistency
- Add comments on GitHub PR page
- Click **"Approve"** when confirmed no issues

### Step 6: Merging to develop

**After PR passes review:**

1. Click **"Merge pull request"** on GitHub PR page
2. Select merge method:
   - **Create a merge commit** (recommended) - preserve complete history
   - Squash and merge - compress into single commit (suitable for many trivial commits)
3. Confirm merge
4. Delete remote feature branch (GitHub will prompt)

**Local Cleanup:**
```bash
# 1. Switch to develop and update
git checkout develop
git pull origin develop

# 2. Delete local feature branch
git branch -d feature/<your-branch-name>

# 3. If remote branch wasn't deleted, manually delete
git push origin --delete feature/<your-branch-name>
```

---

## Commit Message Conventions

### Format
```
<type>(<scope>): <subject>

<body>(optional)
```

### Type Categories
- **feat**: New feature
- **fix**: Bug fix
- **test**: Add/modify tests
- **docs**: Documentation update
- **refactor**: Refactoring (no functionality change)
- **style**: Code formatting (no logic change)
- **chore**: Build/tool configuration

### Scope Categories
- **data**: Data module
- **indexing**: Indexing module
- **search**: Search module
- **baselines**: Baselines module
- **evaluation**: Evaluation module
- **config**: Configuration files
- **docs**: Documentation

### Examples
```bash
# Good commit messages ✅
git commit -m "feat(data): implement query generator with selectivity control"
git commit -m "fix(indexing): correct IVF list assignment boundary check"
git commit -m "test(search): add unit tests for pruning logic"
git commit -m "docs(api): update signature format in API_CONTRACT.md"

# Bad commit messages ❌
git commit -m "update code"
git commit -m "fix bug"
git commit -m "wip"
```

---

## Common Scenarios

### Scenario 1: Forgot to Create Branch from develop

```bash
# If you developed on main branch
git checkout -b feature/temp-save  # Save current work first
git checkout develop
git pull origin develop
git checkout -b feature/correct-branch
git merge feature/temp-save  # Merge previous work
git branch -d feature/temp-save  # Delete temp branch
```

### Scenario 2: Merge Conflicts

```bash
git merge develop
# Conflict prompt appears

# 1. View conflicted files
git status

# 2. Open conflicted files, manually resolve (remove <<<<< ===== >>>>> markers)
# 3. Mark as resolved
git add <resolved-file>

# 4. Complete merge
git commit -m "merge: resolve conflicts with develop"
```

### Scenario 3: Need to Urgently Fix Someone's Bug

```bash
# 1. Create hotfix branch from develop
git checkout develop
git pull origin develop
git checkout -b hotfix/fix-critical-issue

# 2. Fix and commit
git add <fixed-file>
git commit -m "fix(module): resolve critical issue in XXX"

# 3. Push and create PR (mark as urgent)
git push origin hotfix/fix-critical-issue
```

### Scenario 4: Want to Undo Recent Commit

```bash
# Undo commit but keep changes (most common)
git reset --soft HEAD~1

# Undo commit and discard changes (dangerous!)
git reset --hard HEAD~1

# If already pushed, don't use reset, use revert
git revert HEAD
git push origin feature/<branch-name>
```

### Scenario 5: Pull Shows "Need to commit first"

```bash
# Method 1: Stash current changes
git stash
git pull origin develop
git stash pop  # Restore changes

# Method 2: Commit first
git add .
git commit -m "wip: save work in progress"
git pull origin develop
```

---

## Merging to main (Milestone Release)

**Only execute at important milestones (e.g., Phase 1 completion, before final submission):**

```bash
# 1. Ensure develop is stable (all tests pass)
git checkout develop
pytest tests/

# 2. Create PR on GitHub: develop → main
# 3. Team collective review
# 4. After merge, tag the release
git checkout main
git pull origin main
git tag -a v1.0 -m "Phase 1 milestone: all modules implemented"
git push origin v1.0
```

---

## Team Collaboration Best Practices

### ✅ Recommended Practices
1. **Every day before starting work**: `git pull origin develop`
2. **After completing each feature**: commit and push
3. **Before preparing to merge**: merge develop to your branch first, resolve conflicts
4. **Detailed PR descriptions**: let reviewers quickly understand what you did
5. **Respond to reviews promptly**: reply within 24 hours of receiving comments

### ❌ Practices to Avoid
1. **Don't develop on main branch**
2. **Don't go too long without pushing** (more than 2 days)
3. **Don't commit large files** (exclude data files, model files with .gitignore)
4. **Don't force push to shared branches** (develop/main)
5. **Don't ignore conflicts and merge directly**

---

## Checklist

### Before Creating PR
- [ ] Code passes all tests (`pytest tests/`)
- [ ] Function signatures match `docs/API_CONTRACT.md`
- [ ] Added necessary unit tests
- [ ] No sensitive information committed (keys, personal paths)
- [ ] No large files committed (>10MB)
- [ ] commit messages follow conventions
- [ ] All merge conflicts resolved

### Before Merging PR
- [ ] At least 1 person approved
- [ ] CI tests pass (if configured)
- [ ] No unresolved comments
- [ ] Confirm merge target is develop (not main)

---

## Quick Reference

```bash
# Common command reference
git status                    # View status
git log --oneline -10         # View last 10 commits
git branch -a                 # View all branches
git diff                      # View unstaged changes
git diff --staged             # View staged changes

# Branch operations
git checkout <branch>         # Switch branch
git checkout -b <branch>      # Create and switch branch
git branch -d <branch>        # Delete local branch
git push origin --delete <b>  # Delete remote branch

# Sync operations
git pull origin develop       # Pull latest develop code
git push origin <branch>      # Push to remote
git fetch origin              # Fetch remote updates (no merge)

# Undo operations
git checkout -- <file>        # Undo file changes
git reset HEAD <file>         # Unstage
git reset --soft HEAD~1       # Undo last 1 commit
```

---

## Troubleshooting

### Issue 1: Push Rejected
```
Error: ! [rejected] feature/xxx -> feature/xxx (non-fast-forward)
```
**Solution**: Pull first, resolve conflicts, then push
```bash
git pull origin feature/<branch-name>
git push origin feature/<branch-name>
```

### Issue 2: Branch Not Found
```
Error: pathspec 'develop' did not match any file(s) known to git
```
**Solution**: Fetch remote branches
```bash
git fetch origin
git checkout develop
```

### Issue 3: Too Many Conflicts to Resolve
```bash
# Abort current merge
git merge --abort

# Start fresh (use with caution)
git reset --hard origin/develop
```

---

## Team Contact

When encountering Git issues you can't resolve:
1. Consult this document first
2. Ask in team chat
3. Organize online meeting if necessary to resolve collectively

**Important Reminder**: Consult the team first before performing uncertain operations (especially reset, force push)!
