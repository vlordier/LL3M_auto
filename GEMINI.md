# Gemini Development Guide for LL3M

This document provides instructions for Gemini on how to contribute to the LL3M project. Please follow these guidelines for code quality, testing, and general development.

## 1. Key Commands (via `make`)

Use the `Makefile` for common tasks.

- `make check`: Run all checks (linting, type-checking, and tests). **Run this before committing.**
- `make lint`: Run Ruff linter and MyPy type checker.
- `make format`: Format code using `ruff format`.
- `make test`: Run all tests (unit and integration).
- `make test-unit`: Run only unit tests.
- `make test-integration`: Run only integration tests.
- `make test-cov`: Run tests and generate a coverage report. The minimum coverage is 80%.
- `make ci`: Run the full CI pipeline locally.

## 2. Code Quality & Style

### Linter & Formatter: Ruff
- **Check:** `ruff check src/ tests/`
- **Fix:** `ruff check --fix src/ tests/`
- **Format:** `ruff format src/ tests/`
- Configuration is in `pyproject.toml`.

### Type Checking: MyPy
- **Check:** `mypy .` (or use `make lint`)
- Configuration is in `pyproject.toml`.
- Use strict type hinting for all new code.

### Docstrings
- Use **Google Style** for all docstrings.
- Example:
  ```python
  def my_function(arg1: str, arg2: int) -> bool:
      """Short description of the function.

      Longer description explaining what it does.

      Args:
          arg1: Description of the first argument.
          arg2: Description of the second argument.

      Returns:
          Description of the return value.
      """
      # ...
      return True
  ```

## 3. Testing with Pytest

- Test files are located in `tests/`.
- The structure is `tests/unit`, `tests/integration`, `tests/e2e`.
- Use `conftest.py` for shared fixtures.
- **Run all tests:** `pytest` or `make test`
- **Run tests with coverage:** `pytest --cov=src` or `make test-cov`. Minimum coverage is 80%.

## 4. Git Workflow

### Branch Naming
- `feature/<description>`
- `fix/<description>`
- `refactor/<description>`
- `docs/<description>`
- `test/<description>`

### Commit Message Format
Follow the Conventional Commits specification.

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

- **Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `ci`.
- **Scope:** The part of the codebase affected (e.g., `agents`, `blender`, `workflow`).

**Example:**
```
feat(agents): implement critic agent for code verification

- Add CriticAgent to review generated Blender Python code.
- The agent checks for common errors and security vulnerabilities.
- Integration with the main workflow graph.
```

## 5. Project Structure Overview

- `src/agents/`: Core agent implementations (Planner, Coder, Critic, etc.).
- `src/workflow/`: LangGraph state machine and workflow definition.
- `src/blender/`: Blender-specific code, including the executor for running scripts.
- `src/knowledge/`: Integration with external services or knowledge bases.
- `src/utils/`: Shared utilities and type definitions.
- `config/`: Configuration files for agents and other components.
- `tests/`: All tests, separated into `unit`, `integration`, and `e2e`.

By following these guidelines, you will help maintain the quality and consistency of the LL3M codebase.
