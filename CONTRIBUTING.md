# Contributing to Splinator

Thank you for your interest in contributing to Splinator! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/splinator.git
   cd splinator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to all public functions and classes (NumPy style)
- Maximum line length: 120 characters

## Testing

Run tests before submitting PR:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=splinator tests/
```

## Type Checking

Run mypy for type checking:
```bash
mypy src/splinator
```

## Submitting Changes

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear messages
   ```bash
   git commit -m "Add feature: brief description"
   ```

3. Push to your fork and create a Pull Request

## Pull Request Guidelines

- Include tests for new functionality
- Update documentation if needed
- Ensure all tests pass
- Add a clear description of changes
- Reference any related issues

## Reporting Issues

- Use GitHub Issues to report bugs
- Include Python version and minimal reproducible example
- Describe expected vs actual behavior

## Questions?

Feel free to open an issue for any questions about contributing! 