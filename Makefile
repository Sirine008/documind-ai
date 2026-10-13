# DocuMind AI developer commands (Linux/macOS/WSL). Windows PowerShell equivalents are in the README.
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
BACKEND = cd backend &&

.PHONY: help venv install install-ml migrate backend frontend seed eval test test-unit lint format build docker-up docker-down docker-seed

help:
	@echo "install     Create .venv and install backend + frontend dependencies"
	@echo "install-ml  Install CPU torch + sentence-transformers (real embeddings)"
	@echo "migrate     Apply database migrations"
	@echo "backend     Run the API on http://localhost:8000"
	@echo "frontend    Run the UI on http://localhost:5173"
	@echo "seed        Load and index the sample documents"
	@echo "eval        Run the evaluation dataset"
	@echo "test        Run all backend tests (needs PostgreSQL)"
	@echo "test-unit   Run tests that need no database"
	@echo "lint        Ruff + ESLint + TypeScript checks"
	@echo "docker-up   Build and start the full stack with Docker Compose"

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -r backend/requirements-dev.txt
	cd frontend && npm ci

install-ml:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r backend/requirements-ml.txt

migrate:
	$(BACKEND) ../$(PY) -m alembic upgrade head

backend:
	$(BACKEND) ../$(PY) -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

seed:
	$(PY) scripts/seed_demo.py

eval:
	$(PY) scripts/evaluate_rag.py

test:
	$(BACKEND) ../$(PY) -m pytest

test-unit:
	$(BACKEND) ../$(PY) -m pytest -m "not db"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	cd frontend && npm run lint && npm run typecheck

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

build:
	cd frontend && npm run build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-seed:
	docker compose exec backend python /srv/documind/scripts/seed_demo.py
