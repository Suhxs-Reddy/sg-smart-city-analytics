.PHONY: help install test lint format collect detect track analyze serve docker-up docker-down clean

# ── Default ─────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────
install: ## Install all dependencies
	pip install -r requirements.txt
	pip install ruff pre-commit
	@echo "✅ Dependencies installed"

# ── Quality ─────────────────────────────────────
test: ## Run test suite
	PYTHONPATH=. python -m pytest tests/ -v --tb=short

test-fast: ## Run tests (fail on first error)
	PYTHONPATH=. python -m pytest tests/ -v --tb=short -x

lint: ## Lint code with Ruff
	ruff check src/ tests/
	ruff format src/ tests/ --check

format: ## Auto-format code
	ruff format src/ tests/
	ruff check src/ tests/ --fix

# ── Pipeline Stages ─────────────────────────────
collect: ## Run data collection (default: 6 hours)
	PYTHONPATH=. python -m src.ingestion.collector --duration 6

collect-test: ## Quick 6-minute collection test
	PYTHONPATH=. python -m src.ingestion.collector --duration 0.1

detect: ## Run detection on collected data
	PYTHONPATH=. python -m src.pipeline --mode detect

track: ## Run tracking on detected data
	PYTHONPATH=. python -m src.pipeline --mode track

analyze: ## Run analytics on tracked data
	PYTHONPATH=. python -m src.pipeline --mode analyze

pipeline: ## Run full pipeline (detect → track → analyze)
	PYTHONPATH=. python -m src.pipeline --mode full

# ── Serving ─────────────────────────────────────
serve: ## Start API server
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Start API server (production, no reload)
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4

# ── Docker ──────────────────────────────────────
docker-up: ## Start all services (API + collector + MLflow)
	docker compose up -d --build

docker-down: ## Stop all services
	docker compose down

docker-logs: ## View service logs
	docker compose logs -f

# ── Cleanup ─────────────────────────────────────
clean: ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache build dist *.egg-info
	@echo "✅ Cleaned"
