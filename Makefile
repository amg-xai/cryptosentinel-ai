.PHONY: help install lint format typecheck test test-bench audit clean run-api run-dashboard docker-up docker-down

help:
	@echo ""
	@echo "CryptoSentinel AI — available commands"
	@echo "--------------------------------------"
	@echo "  make install      Install all dependencies"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run black formatter"
	@echo "  make typecheck    Run mypy type checker"
	@echo "  make test         Run tests with coverage"
	@echo "  make test-bench   Run benchmark tests"
	@echo "  make audit        Run pip-audit security scan"
	@echo "  make clean        Remove cache and build artifacts"
	@echo "  make docker-up    Start all services via docker-compose"
	@echo "  make docker-down  Stop all services"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check src/ config/ tests/

format:
	black src/ config/ tests/
	ruff check --fix src/ config/ tests/

typecheck:
	mypy config/ src/

test:
	pytest tests/ -v --ignore=tests/benchmarks --ignore=tests/load
test-bench:
	pytest tests/benchmarks/ -v --benchmark-sort=mean

audit:
	pip-audit --output json -o pip-audit-report.json
	@echo "Audit complete. See pip-audit-report.json"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	@echo "Clean complete"

docker-up:
	docker compose up -d

docker-down:
	docker compose down
