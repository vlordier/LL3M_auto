# Good Practices for LL3M Development

This document outlines the development best practices, coding standards, and tooling guidelines for the LL3M project. Following these practices ensures code quality, maintainability, and team collaboration efficiency.

## Table of Contents
- [Code Quality Tools](#code-quality-tools)
- [Project Structure](#project-structure)
- [Type Safety](#type-safety)
- [Testing Practices](#testing-practices)
- [Git Workflow](#git-workflow)
- [Documentation](#documentation)
- [Performance Guidelines](#performance-guidelines)
- [Security Practices](#security-practices)

---

## Code Quality Tools

### 1. Ruff - Fast Python Linter and Formatter

**Configuration in `pyproject.toml`:**
```toml
[tool.ruff]
target-version = "py312"
line-length = 120
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
    "ARG", # flake8-unused-arguments
    "C90", # mccabe complexity
    "N",  # pep8-naming
    "D",  # pydocstyle
    "S",  # flake8-bandit (security)
]
ignore = [
    "D100", # Missing docstring in public module
    "D104", # Missing docstring in public package
    "D211", # No blank lines allowed before class docstring
    "D213", # Multi-line docstring summary should start at the second line
]

[tool.ruff.per-file-ignores]
"tests/*" = ["D", "S101"]  # Ignore docstring and assert rules in tests

[tool.ruff.mccabe]
max-complexity = 10

[tool.ruff.pydocstyle]
convention = "google"
```

**Usage:**
```bash
# Check code
ruff check src/ tests/

# Fix automatically fixable issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### 2. Pre-commit Hooks

**Configuration in `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.10
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.17.1
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
        args: [--ignore-missing-imports, --disallow-untyped-defs, --python-version=3.12]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.6
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        exclude: ^tests/

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [--tb=short, -q]
```

**Setup:**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### 3. MyPy - Static Type Checking

**Configuration in `pyproject.toml`:**
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_optional = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
disallow_untyped_calls = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true
show_error_codes = true
show_column_numbers = true

# Per-module options
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = "bpy.*"
ignore_missing_imports = true
```

**Best Practices:**
- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `typing.TYPE_CHECKING` for import cycles
- Prefer `list[str]` over `List[str]` (Python 3.12+)

**Example:**
```python
from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .types import WorkflowState

async def process_data(
    input_data: Dict[str, Any],
    output_path: Path,
    config: Optional[Dict[str, Any]] = None
) -> tuple[bool, str]:
    """Process data with proper type hints."""
    # Implementation
    return True, "Success"
```

---

## Project Structure

### Modular Architecture Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Use dependency injection for testability
3. **Interface Segregation**: Define clear interfaces/protocols
4. **Layered Architecture**: Separate concerns into distinct layers

**Recommended Structure:**
```
src/
├── agents/          # Agent implementations
│   ├── base.py      # Abstract base classes
│   ├── interfaces.py # Protocols/interfaces
│   └── implementations/
├── workflow/        # LangGraph workflow logic
├── knowledge/       # External service integrations
├── blender/         # Blender-specific functionality
├── utils/           # Shared utilities
├── types/           # Type definitions
└── config/          # Configuration management
```

### Example Interface Definition:
```python
from typing import Protocol, runtime_checkable
from abc import abstractmethod

@runtime_checkable
class Agent(Protocol):
    """Protocol for all LL3M agents."""

    @abstractmethod
    async def process(self, state: 'WorkflowState') -> 'AgentResponse':
        """Process workflow state and return response."""
        ...

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return agent type identifier."""
        ...
```

---

## Type Safety

### 1. Use Pydantic for Data Validation

```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    """Represents a workflow task."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., min_length=1, max_length=100)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: int = Field(default=1, ge=1, le=5)
    dependencies: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('ID must be alphanumeric')
        return v
```

### 2. Use TypedDict for Structured Dictionaries

```python
from typing import TypedDict, Optional, List

class BlenderScriptResult(TypedDict):
    success: bool
    asset_path: Optional[str]
    screenshot_path: Optional[str]
    errors: List[str]
    execution_time: float
```

### 3. Generic Types for Reusability

```python
from typing import TypeVar, Generic, List, Optional

T = TypeVar('T')

class Repository(Generic[T]):
    """Generic repository pattern."""

    def __init__(self) -> None:
        self._items: List[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get_by_id(self, item_id: str) -> Optional[T]:
        # Implementation depends on T having an 'id' attribute
        for item in self._items:
            if hasattr(item, 'id') and item.id == item_id:
                return item
        return None
```

---

## Testing Practices

### 1. Project Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_agents/
│   ├── test_workflow/
│   └── test_utils/
├── integration/             # Integration tests
│   ├── test_blender_integration/
│   └── test_workflow_integration/
├── e2e/                     # End-to-end tests
└── fixtures/                # Test data
    ├── sample_prompts.json
    └── expected_outputs/
```

### 2. Pytest Configuration

**In `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-report=xml",
    "--cov-fail-under=80",
    "--durations=10",
    "-ra",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "e2e: marks tests as end-to-end tests",
    "blender: marks tests that require Blender",
]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
```

### 3. Testing Best Practices

**Fixture Examples:**
```python
# conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

@pytest.fixture
def sample_workflow_state():
    """Sample workflow state for testing."""
    from src.utils.types import WorkflowState
    return WorkflowState(
        prompt="Create a red cube",
        subtasks=[],
        documentation="",
        generated_code="",
    )

@pytest.fixture
def mock_blender_executor():
    """Mock Blender executor."""
    executor = MagicMock()
    executor.execute_code = AsyncMock()
    return executor

@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory):
    """Temporary output directory for tests."""
    return tmp_path_factory.mktemp("ll3m_test_outputs")
```

**Test Examples:**
```python
# test_agents/test_planner.py
import pytest
from unittest.mock import AsyncMock
from src.agents.planner import PlannerAgent
from src.utils.types import WorkflowState, AgentType

class TestPlannerAgent:
    """Test suite for PlannerAgent."""

    @pytest.fixture
    def planner(self):
        return PlannerAgent()

    @pytest.mark.asyncio
    async def test_process_simple_prompt(self, planner, sample_workflow_state):
        """Test processing a simple prompt."""
        sample_workflow_state.prompt = "Create a blue sphere"

        response = await planner.process(sample_workflow_state)

        assert response.success
        assert response.agent_type == AgentType.PLANNER
        assert len(response.data) > 0  # Should return subtasks

    @pytest.mark.asyncio
    async def test_process_complex_prompt(self, planner):
        """Test processing a complex multi-object prompt."""
        state = WorkflowState(
            prompt="Create a scene with a house, trees, and a car with realistic materials"
        )

        response = await planner.process(state)

        assert response.success
        assert len(response.data) >= 3  # House, trees, car

        # Verify subtask types
        subtask_types = [task.type for task in response.data]
        assert 'geometry' in subtask_types
        assert 'material' in subtask_types

    @pytest.mark.parametrize("prompt,expected_subtask_count", [
        ("Create a cube", 1),
        ("Create a red cube with metal material", 2),
        ("Create a scene with multiple objects and lighting", 4),
    ])
    async def test_subtask_generation(self, planner, prompt, expected_subtask_count):
        """Test subtask generation for different prompts."""
        state = WorkflowState(prompt=prompt)
        response = await planner.process(state)

        assert response.success
        assert len(response.data) >= expected_subtask_count
```

**Integration Test Example:**
```python
# integration/test_workflow_integration.py
import pytest
from src.workflow.graph import create_ll3m_workflow
from src.utils.types import WorkflowState

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_simple_object(temp_output_dir):
    """Test complete workflow for simple object creation."""
    workflow = create_ll3m_workflow()

    initial_state = WorkflowState(
        prompt="Create a red cube"
    )

    # Run workflow
    final_state = await workflow.ainvoke(initial_state)

    assert final_state.asset_metadata is not None
    assert final_state.asset_metadata.file_path.exists()
    assert final_state.execution_result.success
    assert len(final_state.detected_issues) == 0
```

### 4. Coverage Guidelines

- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95% coverage for core workflow logic
- **Exception Handling**: Test all error conditions
- **Edge Cases**: Test boundary conditions and invalid inputs

**Coverage Commands:**
```bash
# Run tests with coverage
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html

# Check coverage thresholds
pytest --cov=src --cov-fail-under=80
```

---

## Git Workflow

### 1. Branch Naming Convention

```bash
# Feature branches
feature/agent-implementation
feature/blender-integration

# Bug fixes
fix/memory-leak-in-executor
fix/type-error-in-workflow

# Phase branches
phase-1/foundation-setup
phase-2/agent-development

# Release branches
release/v0.1.0
```

### 2. Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `style`: Code style changes
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```bash
feat(agents): implement planner agent with task decomposition

- Add PlannerAgent class with async processing
- Implement subtask generation from natural language
- Add comprehensive test suite with 90% coverage
- Support complex multi-object scene planning

Closes #123

fix(blender): handle timeout errors in code execution

The Blender executor was not properly handling timeout errors,
causing the workflow to hang indefinitely.

- Add proper timeout handling with asyncio.wait_for
- Improve error messages for timeout scenarios
- Add integration tests for timeout conditions

Fixes #456
```

### 3. Pull Request Template

Create `.github/pull_request_template.md`:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Coverage threshold maintained (80%+)

## Code Quality
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Pre-commit hooks pass
- [ ] Type checking passes (mypy)

## Checklist
- [ ] Branch is up to date with main
- [ ] Commit messages follow conventional format
- [ ] Documentation updated (if needed)
- [ ] Performance impact considered
```

---

## Documentation

### 1. Docstring Standards (Google Style)

```python
def execute_blender_code(
    code: str,
    asset_name: str = "asset",
    timeout: float = 300.0
) -> ExecutionResult:
    """Execute Python code in Blender environment.

    This function wraps the provided Python code with necessary Blender
    setup and teardown logic, executes it in a headless Blender instance,
    and returns the execution result including any generated assets.

    Args:
        code: The Python code to execute in Blender.
        asset_name: Name for the generated asset file.
        timeout: Maximum execution time in seconds.

    Returns:
        ExecutionResult containing success status, file paths, and logs.

    Raises:
        BlenderNotFoundError: When Blender executable is not found.
        ExecutionTimeoutError: When execution exceeds timeout.

    Example:
        >>> executor = BlenderExecutor()
        >>> result = await executor.execute_code(
        ...     "bpy.ops.mesh.primitive_cube_add()",
        ...     asset_name="my_cube"
        ... )
        >>> assert result.success
        >>> assert result.asset_path.exists()
    """
    # Implementation
```

### 2. README Structure

```markdown
# LL3M: Large Language 3D Modelers

Brief project description and key features.

## Installation
Quick start guide

## Usage
Basic usage examples

## Architecture
High-level architecture overview

## Development
Development setup and contribution guidelines

## License
License information
```

### 3. API Documentation

Use tools like `mkdocs` or `sphinx` for comprehensive API documentation:

```yaml
# mkdocs.yml
site_name: LL3M Documentation
nav:
  - Home: index.md
  - Installation: installation.md
  - Usage: usage.md
  - API Reference: api/
  - Development: development.md

theme:
  name: material

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
```

---

## Performance Guidelines

### 1. Async/Await Best Practices

```python
import asyncio
from typing import List

# Good: Concurrent execution
async def process_multiple_tasks(tasks: List[str]) -> List[Result]:
    """Process multiple tasks concurrently."""
    return await asyncio.gather(*[
        process_single_task(task) for task in tasks
    ])

# Bad: Sequential execution
async def process_multiple_tasks_slow(tasks: List[str]) -> List[Result]:
    """Process tasks sequentially (slower)."""
    results = []
    for task in tasks:
        result = await process_single_task(task)
        results.append(result)
    return results
```

### 2. Resource Management

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def blender_session():
    """Manage Blender process lifecycle."""
    process = await start_blender_process()
    try:
        yield process
    finally:
        await process.terminate()
        await process.wait()

# Usage
async def execute_code(code: str):
    async with blender_session() as blender:
        return await blender.execute(code)
```

### 3. Caching and Memoization

```python
from functools import lru_cache
import asyncio

# For sync functions
@lru_cache(maxsize=128)
def expensive_computation(input_data: str) -> str:
    """Cache expensive computations."""
    return process_data(input_data)

# For async functions
class AsyncLRUCache:
    def __init__(self, maxsize: int = 128):
        self.cache = {}
        self.maxsize = maxsize

    async def get_or_compute(self, key: str, compute_func):
        if key not in self.cache:
            if len(self.cache) >= self.maxsize:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[key] = await compute_func(key)

        return self.cache[key]
```

---

## Security Practices

### 1. Environment Variables

```python
import os
from pathlib import Path

# Good: Use environment variables for secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Bad: Hardcoded secrets
OPENAI_API_KEY = "sk-1234567890abcdef"  # Never do this!
```

### 2. Input Validation

```python
from pydantic import BaseModel, Field, field_validator
import re

class UserPrompt(BaseModel):
    """Validated user prompt."""

    text: str = Field(..., min_length=1, max_length=1000)

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        # Remove potentially dangerous content
        if re.search(r'import\s+os|subprocess|eval|exec', v, re.IGNORECASE):
            raise ValueError("Prompt contains potentially unsafe content")
        return v.strip()
```

### 3. Safe Code Execution

```python
import ast
from typing import Set

class SafeCodeValidator:
    """Validate that generated code is safe to execute."""

    FORBIDDEN_MODULES = {
        'os', 'subprocess', 'sys', 'importlib',
        'eval', 'exec', 'compile', '__import__'
    }

    def validate_code(self, code: str) -> bool:
        """Check if code is safe to execute."""
        try:
            tree = ast.parse(code)
            return self._check_ast_node(tree)
        except SyntaxError:
            return False

    def _check_ast_node(self, node: ast.AST) -> bool:
        """Recursively check AST nodes for forbidden operations."""
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name in self.FORBIDDEN_MODULES:
                        return False
            elif isinstance(child, ast.ImportFrom):
                if child.module in self.FORBIDDEN_MODULES:
                    return False
        return True
```

---

## Make Commands Reference

```bash
# Setup and installation
make install          # Install production dependencies
make dev             # Install development dependencies
make setup-blender   # Setup Blender for development

# Code quality
make lint            # Run all linters (ruff, mypy)
make format          # Format code (ruff format)
make check           # Run all checks (lint + test)

# Testing
make test            # Run all tests
make test-unit       # Run unit tests only
make test-integration # Run integration tests only
make test-cov        # Run tests with coverage report
make test-watch      # Run tests in watch mode

# Development
make clean           # Clean up generated files
make docs            # Generate documentation
make dev-server      # Start development server
make build           # Build distribution packages

# CI/CD
make ci              # Run full CI pipeline locally
make pre-commit      # Run pre-commit hooks
```

---

## Checklist for New Features

Before submitting any new feature:

### Code Quality
- [ ] All functions have type hints
- [ ] Docstrings follow Google style
- [ ] Code passes ruff checks
- [ ] MyPy type checking passes
- [ ] Pre-commit hooks pass

### Testing
- [ ] Unit tests written with 90%+ coverage
- [ ] Integration tests for complex workflows
- [ ] Edge cases and error conditions tested
- [ ] Performance tests for critical paths

### Documentation
- [ ] API documentation updated
- [ ] Usage examples provided
- [ ] Breaking changes documented
- [ ] Migration guide (if needed)

### Performance
- [ ] Async/await used appropriately
- [ ] Resource cleanup implemented
- [ ] Memory usage considered
- [ ] Caching strategy evaluated

### Security
- [ ] Input validation implemented
- [ ] No hardcoded secrets
- [ ] Safe execution practices followed
- [ ] Error messages don't leak sensitive info

Following these practices ensures high-quality, maintainable, and secure code that scales well as the project grows.
