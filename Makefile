.PHONY: run test install setup

# Install dependencies
install:
	pip install -r requirements.txt

# Setup: create .env from example if it doesn't exist
setup:
	@test -f .env || cp .env.example .env
	@echo "Environment file ready. Edit .env to customize settings."

# Run the API server
run:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Run the demo example
demo:
	python -m examples.demo

# Format code
format:
	black backend/ tests/ examples/
	isort backend/ tests/ examples/

# Lint
lint:
	ruff check backend/ tests/
