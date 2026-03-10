# Git Strategy — Singapore Smart City Analytics

**Dual-Track Development: Production (main) + Research (feature branches)**

---

## Branch Strategy

```
main (production-ready, always deployable)
├── docs/update-readme                    # Documentation improvements
├── feature/dashboard                     # Visual dashboard
├── feature/research-self-supervised      # Novel research track ⭐
└── hotfix/api-performance                # Production bug fixes
```

**Golden Rule**: `main` is ALWAYS deployable. Never push broken code.

---

## Branch Policies

### `main` (Production)
- ✅ Always passes CI/CD (tests, lint, security)
- ✅ Always deployable to Azure
- ✅ Stable baseline model (85% mAP)
- ✅ Protected branch (no force push)
- 🔄 Auto-deploys to Azure on push
- 📊 GitHub Actions badges must be green

**Merge Requirements**:
- All tests pass
- Code review (self-review with detailed commit messages)
- No merge conflicts

### `feature/research-self-supervised` (Research)
- 🔬 Experimental research code
- 🔬 Can break, iterate freely
- 🔬 Separate CI/CD (tests but doesn't deploy)
- 📝 Detailed experiment tracking in commit messages
- 📊 MLflow runs logged with branch name

**Merge to main when**:
- Research proves successful (>90% mAP with 100 labels)
- All tests pass
- Documentation updated
- Results reproducible

---

## Commit Message Convention

**Format**: `type(scope): description`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `research`: Research experiments
- `deploy`: Deployment changes
- `ci`: CI/CD changes

**Examples**:
```bash
# Production commits
feat(api): add camera health check endpoint
fix(detector): handle CUDA out of memory errors
deploy(azure): optimize Docker image size

# Research commits
research(ssl): implement MoCo v3 pre-training
research(ssl): experiment with temperature=0.07 vs 0.1
research(ssl): baseline results - 65% mAP with 100 labels
research(ssl): pre-trained model - 90% mAP with 100 labels ⭐
```

---

## Workflow

### Week 1: Production Track (main)

```bash
# Day 1: Push current infrastructure
git checkout main
git add -A
git commit -m "feat: complete production infrastructure

- GitHub Actions CI/CD (5 workflows)
- Azure deployment automation
- Production-grade training notebook
- Comprehensive documentation
- 80+ tests passing

Status: Ready for deployment"

git push origin main
# → Triggers CI/CD, all badges turn green

# Day 1-2: Deploy and test
# (No code changes, just running deployment scripts)

# Day 2: Document deployment
git checkout -b docs/deployment-results
# Take screenshots, record metrics
git add docs/screenshots/
git commit -m "docs: add deployment results and metrics

- Live API endpoint: http://20.205.xxx.xxx:8000
- Baseline model: 85% mAP on UA-DETRAC
- Inference speed: 110 FPS on T4
- System uptime: 99.9%
- Screenshots of working dashboard"

git push origin docs/deployment-results
# Create PR, merge to main after review
```

### Week 2-3: Research Track (feature branch)

```bash
# Day 3: Create research branch
git checkout -b feature/research-self-supervised

# Day 3-10: Iterative research
git commit -m "research(ssl): initialize MoCo v3 implementation"
git commit -m "research(ssl): add Singapore dataset loader"
git commit -m "research(ssl): pre-train on 10k images - initial test"
git commit -m "research(ssl): pre-train on 100k images - full run"
git commit -m "research(ssl): fine-tune with 100 labels - 68% mAP"
git commit -m "research(ssl): fine-tune with 100 labels - 90% mAP ⭐"

# Push research branch (doesn't affect production)
git push origin feature/research-self-supervised

# Day 10-12: Validate and document
git commit -m "research(ssl): reproduce results 3x - consistent 90% mAP"
git commit -m "docs(research): add research paper draft"
git commit -m "test(ssl): add unit tests for pre-training pipeline"

# Day 12: Merge to main (research proven successful)
git checkout main
git merge feature/research-self-supervised
git commit -m "feat: self-supervised pre-training foundation model

NOVEL RESEARCH CONTRIBUTION:
- Implemented MoCo v3 on 100k unlabeled Singapore images
- Achieved 90% mAP with only 100 labeled images
- 25% improvement over baseline (65% → 90%)
- 100x more data-efficient than standard fine-tuning

Results:
- Pre-training: 100k unlabeled images, 48h on T4
- Fine-tuning: 100 labeled images, 2h on T4
- Test mAP: 90.3% (±0.5% over 3 runs)

This is the first Singapore-specific traffic foundation model.

Closes #research-ssl"

git push origin main
```

---

## CI/CD Per Branch

### `main` branch CI/CD
```yaml
# .github/workflows/ci.yml (already created)
on:
  push:
    branches: [ main ]

jobs:
  - lint
  - test
  - docker-build
  - validate-configs
  - security-scan
  - deploy-to-azure  # ← Only on main
```

### Research branches CI/CD
```yaml
# .github/workflows/research-ci.yml (new)
on:
  push:
    branches: [ 'feature/research-*' ]

jobs:
  - lint
  - test
  - validate-configs
  # NO deployment (research doesn't go to production)
  # But can run MLflow tracking
```

---

## Protecting Production

### GitHub Branch Protection Rules

Set these on GitHub (Settings → Branches → Add rule):

**For `main`**:
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Require linear history (no merge commits)
- ✅ Require deployments to succeed before merging
- ❌ Allow force pushes (NEVER on main)

---

## MLflow Integration with Git

Every MLflow run logs git metadata:

```python
# In training notebooks
import mlflow
import git

repo = git.Repo(search_parent_directories=True)
git_commit = repo.head.object.hexsha
git_branch = repo.active_branch.name

mlflow.log_params({
    "git_commit": git_commit,
    "git_branch": git_branch,
    "is_research": "research" in git_branch,
})
```

**Benefit**: Every experiment is tied to exact git commit. Fully reproducible.

---

## Release Strategy

### Semantic Versioning

**Format**: `v<major>.<minor>.<patch>`

**Version History**:
```
v1.0.0 - Production baseline (current)
  - 85% mAP on UA-DETRAC
  - Azure deployment working
  - Dashboard live

v1.1.0 - Domain adaptation (after 1 week data collection)
  - 92% mAP on Singapore data
  - Retrained on collected images

v2.0.0 - Self-supervised foundation model (research)
  - 90% mAP with 100x less labels
  - Novel research contribution
  - arXiv paper submitted
```

### Git Tags

```bash
# Tag production releases
git tag -a v1.0.0 -m "Production baseline - 85% mAP"
git push origin v1.0.0

# Tag research milestones
git tag -a v2.0.0 -m "Foundation model - 90% mAP with 100 labels"
git push origin v2.0.0
```

---

## Commit Frequency

**Production (`main`)**:
- Commit after each logical unit of work
- 1-3 commits per day
- Each commit should pass tests

**Research (`feature/research-*`)**:
- Commit after each experiment
- 5-10 commits per day
- Frequent checkpoints (experiments can fail)

**Example research day**:
```bash
9am:  research(ssl): try temperature=0.05
11am: research(ssl): try temperature=0.07 - better loss
1pm:  research(ssl): try temperature=0.10 - overfitting
3pm:  research(ssl): final config temperature=0.07, lr=0.01
5pm:  research(ssl): full run on 100k images started
```

---

## Code Review (Self-Review for Solo Projects)

Before merging research → main:

**Checklist**:
- [ ] All tests pass
- [ ] Code is documented (docstrings)
- [ ] Results are reproducible (3+ runs)
- [ ] MLflow runs logged
- [ ] README updated
- [ ] Research paper/report drafted
- [ ] Breaking changes documented
- [ ] Performance regression checked

**Self-review process**:
```bash
# Create PR even for solo projects (good practice)
git push origin feature/research-self-supervised

# On GitHub: Create Pull Request
# Title: "Self-supervised pre-training foundation model"
# Description: Detailed results, screenshots, MLflow links
# Review your own code, add comments
# Then merge
```

---

## Disaster Recovery

### If main breaks

```bash
# Option 1: Revert last commit
git revert HEAD
git push origin main

# Option 2: Reset to last known good commit
git reset --hard <good-commit-sha>
git push --force origin main  # Only in emergency!

# Option 3: Hotfix branch
git checkout -b hotfix/fix-deployment-error
# Fix the issue
git commit -m "fix: resolve deployment error"
git push origin hotfix/fix-deployment-error
# Fast-track merge to main
```

### If research experiment fails

```bash
# No problem! Research branch can fail
git checkout feature/research-self-supervised
git reset --hard HEAD~5  # Go back 5 commits
# Try different approach
```

---

## .gitignore Best Practices

Already configured, but verify:

```gitignore
# ML artifacts (never commit)
*.pt
*.onnx
*.pth
mlruns/
wandb/

# Data (too large for git)
data/raw/
data/processed/
*.jpg
*.png

# Credentials (NEVER commit)
.env
credentials.json
*.pem
*.key

# Python
__pycache__/
*.pyc
.venv/

# OS
.DS_Store
```

**If you accidentally commit sensitive data**:
```bash
# Remove from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (only if repo is private!)
git push origin --force --all
```

---

## GitHub Actions Status Checks

**Required checks before merge**:
- ✅ Lint (Ruff)
- ✅ Tests (pytest)
- ✅ Docker build
- ✅ Config validation
- ✅ Security scan

**On `main`**:
- ✅ All above + deployment succeeds

---

## Summary: Healthy Git Workflow

```
Week 1 (Production)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
main (always green)
 ├─ commit: infrastructure complete
 ├─ commit: deployed to Azure
 ├─ commit: baseline model results
 └─ commit: documentation update

Week 2-3 (Research)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
feature/research-self-supervised (experimental)
 ├─ commit: implement MoCo v3
 ├─ commit: pre-train experiment 1
 ├─ commit: pre-train experiment 2
 ├─ commit: pre-train experiment 3
 ├─ commit: fine-tune results - 90% mAP ⭐
 └─ PR → main (research proven, merge)

main (still green, now with research)
 └─ commit: merge research - foundation model
```

**Result**: Clean git history, production stays stable, research can iterate freely.

---

**Next**: Commit current work to main, then create execution runbook.
