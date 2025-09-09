# LL3M: Large Language 3D Modelers

A multi-agent system that generates 3D assets by writing Python code in Blender using large language models. This implementation uses LangGraph for orchestration, OpenAI GPT models for reasoning, and Context7 MCP for Blender documentation retrieval.

## Features

- **Multi-Agent Architecture**: Coordinated agents for planning, retrieval, coding, criticism, and verification
- **Three-Phase Workflow**: Initial creation → Automatic refinement → User-guided refinement
- **LangGraph Orchestration**: State-driven workflow management with conditional branching
- **Context7 Integration**: Real-time Blender API documentation retrieval
- **Async Execution**: High-performance async Blender code execution
- **Type Safety**: Comprehensive type hints and Pydantic models throughout

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Planner   │ → │  Retrieval  │ → │   Coding    │
│    Agent    │    │    Agent    │    │   Agent     │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Verification │ ← │   Critic    │ ← │  Execution  │
│   Agent     │    │   Agent     │    │   Engine    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- Blender 3.0+ (with Python API access)
- OpenAI API key
- Context7 MCP server access

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LL3M_auto
```

2. Install dependencies:
```bash
make dev
```

3. Set up Blender:
```bash
make setup-blender
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and Blender path
```

## Usage

### Basic Usage

```python
from src.workflow.graph import create_ll3m_workflow
from src.utils.types import WorkflowState

# Create workflow
workflow = create_ll3m_workflow()

# Generate 3D asset
initial_state = WorkflowState(prompt="Create a red cube with metallic material")
result = await workflow.ainvoke(initial_state)

print(f"Asset created: {result.asset_metadata.file_path}")
```

### CLI Interface

```bash
# Generate new asset
ll3m generate "a futuristic robot with glowing eyes"

# Refine existing asset (by asset_id or path)
ll3m refine asset_123 "make the robot taller and add wings"

# Export asset to GLTF (by asset_id or path)
ll3m export asset_123 --format gltf --output robot.gltf

Notes:
- `asset_123` can be an asset_id (as recorded in `asset_metadata.json`) or a full path to the asset directory.
- Exports are copied from the asset's `exports/` folder. If none exist, regenerate with export formats enabled or run a dedicated export pipeline.
```

## Development

### Project Structure

```
src/
├── agents/          # Agent implementations
├── knowledge/       # Context7 MCP integration
├── blender/         # Blender execution engine
├── workflow/        # LangGraph workflow definitions
└── utils/           # Utilities and type definitions
```

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run full check
ruff check src/ tests/
mypy src/
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options:

- `OPENAI_API_KEY`: Your OpenAI API key
- `BLENDER_PATH`: Path to Blender executable
- `CONTEXT7_MCP_SERVER`: Context7 MCP server URL
- `OUTPUT_DIRECTORY`: Directory for generated assets

### Agent Configuration

Agent-specific settings can be customized in the configuration files:

```yaml
# config/agents.yaml
planner:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 1000
```

## Phase 1 Status

✅ **Foundation & Setup Complete**
- [x] Project structure established
- [x] Dependencies configured
- [x] Context7 MCP integration
- [x] Basic Blender execution
- [x] Type safety and configuration
- [x] Testing framework (80%+ coverage)
- [x] Development tooling (ruff, mypy, pre-commit)

🚧 **Next Phases**
- [ ] Phase 2: Agent Implementation
- [ ] Phase 3: LangGraph Workflow
- [ ] Phase 4: Advanced Blender Integration
- [ ] Phase 5: User Interface & API
- [ ] Phase 6: Testing & Validation

## Contributing

1. Follow the coding standards in `GOOD_PRACTICES.md`
2. Ensure all tests pass: `make test`
3. Run code quality checks: `make lint format`
4. Submit pull requests with clear descriptions

## License

MIT License - see LICENSE file for details.
