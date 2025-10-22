# Git 工作流程文档

本文档定义了CS6400 Group 11项目的Git协作规范，所有团队成员必须遵守。

## 分支策略

### 分支类型

```
main (生产分支)
  └── develop (开发集成分支)
       ├── feature/data-loader (功能分支 - Yao-Ting)
       ├── feature/indexing-ivf (功能分支 - Zaowei)
       ├── feature/search-pruning (功能分支 - Yichang)
       └── feature/baselines-eval (功能分支 - Zhangding)
```

**分支说明：**
- **main**: 稳定的里程碑版本，只在项目重要节点更新
- **develop**: 开发集成分支，所有功能分支merge到这里
- **feature/\<module\>-\<name\>**: 个人功能分支，从develop创建

### 分支规则

1. ✅ **允许的操作**：
   - 从develop创建feature分支
   - feature分支merge到develop
   - develop在里程碑时merge到main

2. ❌ **禁止的操作**：
   - 直接push到main分支
   - feature分支直接merge到main
   - 跨feature分支直接merge

---

## 完整工作流程

### 步骤1：开始新任务

```bash
# 1. 确保在develop分支
git checkout develop

# 2. 拉取最新代码（避免基于过期代码开发）
git pull origin develop

# 3. 创建功能分支
git checkout -b feature/<module>-<description>

# 示例：
git checkout -b feature/data-loader
git checkout -b feature/indexing-signatures
git checkout -b feature/search-pruning
git checkout -b feature/baseline-postfilter
```

### 步骤2：开发过程中

```bash
# 1. 编写代码...

# 2. 查看修改
git status
git diff

# 3. 添加修改到暂存区
git add <文件名>           # 添加指定文件
git add .                  # 添加所有修改（谨慎使用）

# 4. 提交（遵循commit规范）
git commit -m "feat(data): implement Zipf distribution for category generation"

# 5. 定期推送到远程（备份 + 让团队看到进度）
git push origin feature/<你的分支名>
```

**开发建议：**
- 频繁commit（每完成一个小功能就commit）
- 每天至少push一次（避免代码丢失）
- commit message要清晰描述做了什么

### 步骤3：完成开发，准备合并

```bash
# 1. 确保所有修改已提交
git status  # 应该显示 "nothing to commit, working tree clean"

# 2. 拉取最新develop（可能其他人已经合并了代码）
git checkout develop
git pull origin develop

# 3. 切回feature分支
git checkout feature/<你的分支名>

# 4. 将develop的最新代码合并到你的分支（解决可能的冲突）
git merge develop

# 5. 如果有冲突，解决后提交
git add <解决冲突的文件>
git commit -m "merge: resolve conflicts with develop"

# 6. 推送到远程
git push origin feature/<你的分支名>
```

### 步骤4：创建Pull Request (PR)

**在GitHub网页操作：**

1. 访问：https://github.com/ZhangdingLiu/CS6400_Project_Group11
2. 点击 **"Pull requests"** → **"New pull request"**
3. 设置分支：
   - **base**: `develop` ← 合并目标
   - **compare**: `feature/<你的分支名>` ← 你的功能分支
4. 填写PR标题和描述：

```markdown
## 完成内容
- [x] 实现了XXX功能
- [x] 添加了单元测试
- [x] 更新了相关文档

## 测试情况
- [x] 所有单元测试通过 (`pytest tests/test_xxx.py`)
- [x] 函数签名与API_CONTRACT.md一致
- [x] 代码符合Python规范（类型提示、文档字符串）

## 接口验证
- [x] 输入/输出格式符合API_CONTRACT规范
- [x] 与依赖模块接口匹配

## 注意事项
（说明任何需要reviewer注意的地方）
```

5. 请求团队成员review（右侧Reviewers选择至少1人）
6. 等待review反馈

### 步骤5：代码Review

**作为PR创建者：**
- 根据review意见修改代码
- 在feature分支上commit并push（PR会自动更新）
- 回复reviewer的comments

**作为Reviewer：**
- 检查代码质量、测试覆盖、接口一致性
- 在GitHub PR页面添加comments
- 确认无问题后点击 **"Approve"**

### 步骤6：合并到develop

**PR通过review后：**

1. 在GitHub PR页面点击 **"Merge pull request"**
2. 选择合并方式：
   - **Create a merge commit** (推荐) - 保留完整历史
   - Squash and merge - 压缩为单个commit（适合很多琐碎commit）
3. 确认merge
4. 删除远程feature分支（GitHub会提示）

**本地清理：**
```bash
# 1. 切换到develop并更新
git checkout develop
git pull origin develop

# 2. 删除本地feature分支
git branch -d feature/<你的分支名>

# 3. 如果远程分支没删除，手动删除
git push origin --delete feature/<你的分支名>
```

---

## Commit Message 规范

### 格式
```
<type>(<scope>): <subject>

<body>（可选）
```

### Type类型
- **feat**: 新功能
- **fix**: bug修复
- **test**: 添加/修改测试
- **docs**: 文档更新
- **refactor**: 重构（不改变功能）
- **style**: 代码格式调整（不影响逻辑）
- **chore**: 构建/工具配置

### Scope范围
- **data**: 数据模块
- **indexing**: 索引模块
- **search**: 搜索模块
- **baselines**: 基线模块
- **evaluation**: 评估模块
- **config**: 配置文件
- **docs**: 文档

### 示例
```bash
# 好的commit message ✅
git commit -m "feat(data): implement query generator with selectivity control"
git commit -m "fix(indexing): correct IVF list assignment boundary check"
git commit -m "test(search): add unit tests for pruning logic"
git commit -m "docs(api): update signature format in API_CONTRACT.md"

# 不好的commit message ❌
git commit -m "update code"
git commit -m "fix bug"
git commit -m "wip"
```

---

## 常见场景处理

### 场景1：忘记从develop创建分支

```bash
# 如果你在main分支上开发了
git checkout -b feature/temp-save  # 先保存当前工作
git checkout develop
git pull origin develop
git checkout -b feature/correct-branch
git merge feature/temp-save  # 把之前的工作合并过来
git branch -d feature/temp-save  # 删除临时分支
```

### 场景2：合并时出现冲突

```bash
git merge develop
# 出现冲突提示

# 1. 查看冲突文件
git status

# 2. 打开冲突文件，手动解决（删除<<<<< ===== >>>>>标记）
# 3. 标记为已解决
git add <解决的文件>

# 4. 完成merge
git commit -m "merge: resolve conflicts with develop"
```

### 场景3：需要紧急修复别人的bug

```bash
# 1. 从develop创建hotfix分支
git checkout develop
git pull origin develop
git checkout -b hotfix/fix-critical-issue

# 2. 修复并提交
git add <修复的文件>
git commit -m "fix(module): resolve critical issue in XXX"

# 3. 推送并创建PR（标注为urgent）
git push origin hotfix/fix-critical-issue
```

### 场景4：想撤销最近的commit

```bash
# 撤销commit但保留修改（最常用）
git reset --soft HEAD~1

# 撤销commit且丢弃修改（危险！）
git reset --hard HEAD~1

# 如果已经push，不要用reset，用revert
git revert HEAD
git push origin feature/<分支名>
```

### 场景5：拉取时出现"需要先commit"

```bash
# 方法1：暂存当前修改
git stash
git pull origin develop
git stash pop  # 恢复修改

# 方法2：先commit
git add .
git commit -m "wip: save work in progress"
git pull origin develop
```

---

## 合并到main的流程（里程碑发布）

**仅在重要节点执行（如Phase 1完成、最终提交前）：**

```bash
# 1. 确保develop稳定（所有测试通过）
git checkout develop
pytest tests/

# 2. 在GitHub创建PR: develop → main
# 3. 团队集体review
# 4. Merge后打tag
git checkout main
git pull origin main
git tag -a v1.0 -m "Phase 1 milestone: all modules implemented"
git push origin v1.0
```

---

## 团队协作最佳实践

### ✅ 推荐做法
1. **每天开始工作前**：`git pull origin develop`
2. **每完成一个功能**：commit并push
3. **准备merge前**：先merge develop到自己分支，解决冲突
4. **PR描述详细**：让reviewer快速理解你做了什么
5. **及时响应review**：收到comments后24小时内回复

### ❌ 避免做法
1. **不要在main分支开发**
2. **不要长时间不push**（超过2天）
3. **不要提交大文件**（数据文件、模型文件用.gitignore排除）
4. **不要force push到共享分支**（develop/main）
5. **不要忽略冲突直接merge**

---

## 检查清单

### 创建PR前必查
- [ ] 代码通过所有测试 (`pytest tests/`)
- [ ] 函数签名与 `docs/API_CONTRACT.md` 一致
- [ ] 添加了必要的单元测试
- [ ] 没有提交敏感信息（密钥、个人路径）
- [ ] 没有提交大文件（>10MB）
- [ ] commit message符合规范
- [ ] 已解决所有merge冲突

### Merge PR前必查
- [ ] 至少1人approve
- [ ] CI测试通过（如果配置了）
- [ ] 没有unresolved comments
- [ ] 确认merge目标是develop（不是main）

---

## 快速参考

```bash
# 常用命令速查
git status                    # 查看状态
git log --oneline -10         # 查看最近10条commit
git branch -a                 # 查看所有分支
git diff                      # 查看未暂存的修改
git diff --staged             # 查看已暂存的修改

# 分支操作
git checkout <branch>         # 切换分支
git checkout -b <branch>      # 创建并切换分支
git branch -d <branch>        # 删除本地分支
git push origin --delete <b>  # 删除远程分支

# 同步操作
git pull origin develop       # 拉取develop最新代码
git push origin <branch>      # 推送到远程
git fetch origin              # 获取远程更新（不合并）

# 撤销操作
git checkout -- <file>        # 撤销文件修改
git reset HEAD <file>         # 取消暂存
git reset --soft HEAD~1       # 撤销最近1次commit
```

---

## 问题排查

### 问题1：push被拒绝
```
错误: ! [rejected] feature/xxx -> feature/xxx (non-fast-forward)
```
**解决**：先pull，解决冲突后再push
```bash
git pull origin feature/<分支名>
git push origin feature/<分支名>
```

### 问题2：找不到分支
```
错误: pathspec 'develop' did not match any file(s) known to git
```
**解决**：fetch远程分支
```bash
git fetch origin
git checkout develop
```

### 问题3：冲突太多无法解决
```bash
# 放弃当前merge
git merge --abort

# 重新开始（慎用）
git reset --hard origin/develop
```

---

## 团队联系方式

遇到Git问题无法解决时：
1. 先查阅本文档
2. 在团队群里询问
3. 必要时组织线上会议集体解决

**重要提醒**：不确定的操作（尤其是reset、force push）请先咨询团队！
