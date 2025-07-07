.PHONY: help install install-dev test test-cov lint format type-check docs clean build

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install the package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  docs         Build documentation"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build distribution packages"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

test:
	pytest tests/

test-cov:
	pytest --cov=splinator --cov-report=html --cov-report=term tests/

lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/splinator

docs:
	cd docs && make clean && make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build 