.PHONY: help install test lint format run docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies in virtual environment"
	@echo "  make test          Run automated test suite"
	@echo "  make lint          Run Ruff linter"
	@echo "  make format        Format code with Ruff"
	@echo "  make run           Launch production Web Dashboard (port 8080)"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start container with Docker Compose"
	@echo "  make clean         Clean temporary artifacts and caches"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest ruff httpx

test:
	pytest -v

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

run:
	python main.py --serve --port 8080

docker-build:
	docker build -t cv-intelligence-suite:latest .

docker-up:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
