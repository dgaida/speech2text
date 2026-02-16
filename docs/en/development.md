# Development

## Setting Up the Environment

Follow the instructions in the [Installation](installation.md) section for the development environment.

## Coding Guidelines

- **Docstrings**: All public APIs must have Google-style docstrings.
- **Formatting**: We use `black` with a line length of 127.
- **Linting**: We use `ruff`.
- **Typing**: Strict typing with `mypy` is required.

## Running Tests

We use `pytest` for our test suite.

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=speech2text
```

## Building Documentation Locally

```bash
# Live preview
mkdocs serve

# Build static site
mkdocs build
```
