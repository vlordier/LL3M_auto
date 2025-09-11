# LL3M: Large Language 3D Modelers

A production-ready multi-agent system that generates 3D assets by writing Python code in Blender using large language models. Features a comprehensive web interface with multi-step workflows, structured reformulation, and real-time progress tracking.

## Quick Start

Get LL3M running in 3 steps:

```bash
# 1. Install dependencies
make dev

# 2. Configure (add your OpenAI API key)  
cp .env.example .env && nano .env

# 3. Start web interface
make web-run-openai
# Visit http://localhost:3000
```

Try it out: Type "Create a spaceship" and experience the intelligent reformulation with interactive option pills!

## Features

- **🎯 Multi-Step Workflow**: Interactive reformulation with structured questions and user confirmation
- **🌐 Full Web Interface**: Modern React frontend with FastAPI backend and real-time WebSocket updates
- **🤖 Multi-Agent Architecture**: Coordinated agents for planning, retrieval, coding, criticism, and verification  
- **🔄 Smart Reformulation**: Structured question generation with interactive option pills for precise user intent
- **📊 Real-time Progress**: Live workflow status with detailed progress tracking and session management
- **🧪 Comprehensive Testing**: Full Playwright e2e test suite validating multi-step workflows
- **⚡ High Performance**: Async execution with streamlined codebase and optimized dependencies
- **🛡️ Type Safety**: Complete type coverage with Pydantic models and strict linting

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

### Web Interface (Primary UI)

The full-featured web application with multi-step workflow support:

```bash
# Start both frontend and backend with OpenAI integration
make web-run-openai

# Access the interface
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000  
# API Docs: http://localhost:8000/docs
```

**Multi-Step Workflow Features:**
- 📝 **Smart Reformulation**: Enter a basic prompt like "Create a spaceship" and receive structured questions with interactive options
- 🎛️ **Interactive Options**: Choose from generated option pills like "Futuristic and sleek", "Industrial and rugged", etc.
- ✅ **User Confirmation**: Review enhanced prompts before proceeding with generation
- 📊 **Live Progress**: Real-time updates throughout the generation pipeline
- 🎨 **Rich Visuals**: Enhanced UI with 118+ visual design elements

**API Diagnostics:**
```bash
curl 'http://localhost:8000/health'                    # Health check
curl 'http://localhost:8000/api/diagnostics'          # Basic diagnostics  
curl 'http://localhost:8000/api/diagnostics?full=1'   # Full system check
```

### CLI Interface

```bash
# Generate new asset
ll3m generate "a futuristic robot with glowing eyes"

# Refine existing asset (by asset_id or path)
ll3m refine asset_123 "make the robot taller and add wings"

# Export asset to GLTF (by asset_id or path)
ll3m export asset_123 --format gltf --output robot.gltf

# If no export exists yet, force export from .blend (if available)
ll3m export asset_123 --format gltf --output robot.gltf --force

# Discover assets and inspect
ll3m list-assets --format table
ll3m show asset_123

### Testing & Validation

**Playwright E2E Tests** - Comprehensive multi-step workflow validation:

```bash
# Run full e2e test suite (requires backend running)
cd web/frontend && npm test

# Specific workflow tests
npx playwright test tests/e2e/real-workflow-test.spec.ts        # Backend integration
npx playwright test tests/e2e/improved-reformulation.spec.ts   # Structured questions  
npx playwright test tests/e2e/complete-generation-flow.spec.ts # Full generation flow

# Quick smoke test
make web-smoke
```

**Recent Test Results:**
- ✅ Multi-step reformulation workflow functional
- ✅ 11 interactive question option pills generated and working
- ✅ Structured question components detected and responsive  
- ✅ User confirmation flow with "Use Enhanced Version" button working
- ✅ Real-time progress indicators and session management active

Notes:
- `asset_123` can be an asset_id (as recorded in `asset_metadata.json`) or a full path to the asset directory.
- Exports are copied from the asset's `exports/` folder. If none exist, regenerate with export formats enabled or use `--force` to export directly from a `.blend` file when present.

### Notes on Outputs

The canonical output structure lives under `outputs/projects/.../assets/.../<stage>/<asset_id>`. The tooling provides a lightweight asset index (`outputs/asset_index.json`) for fast lookups by `asset_id`.
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

**Python Tests:**
```bash
# Run all tests (unit + integration)
make test

# Run with coverage reporting
make test-cov

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
```

**Frontend E2E Tests:**
```bash
# Full multi-step workflow validation (requires backend)
cd web/frontend && npm test

# Run with browser UI for debugging
cd web/frontend && npm run test:headed
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

## Project Status

✅ **Phase 4 Complete: Clean Production Deployment**
- [x] **Multi-Agent System**: All agents implemented with LangGraph orchestration
- [x] **Web Interface**: Production-ready React frontend + FastAPI backend
- [x] **Multi-Step Workflow**: Smart reformulation with structured questions
- [x] **Interactive UI**: 11+ question option pills, user confirmation flows
- [x] **Real-time Features**: WebSocket progress tracking, session management
- [x] **Comprehensive Testing**: Playwright e2e tests validating full workflows
- [x] **Code Quality**: Streamlined codebase, 100% type coverage, automated CI/CD
- [x] **Production Ready**: Containerized deployment, health checks, diagnostics

🎯 **Current Capabilities**
- ✨ **Smart Asset Generation**: Natural language → Structured questions → 3D assets
- 🎛️ **Interactive Refinement**: Real-time option selection with visual feedback
- 📊 **Session Management**: Persistent workflows with progress tracking  
- 🧪 **Validated Workflows**: E2E tested multi-step generation pipeline
- ⚡ **High Performance**: Async execution with optimized agent coordination

## Contributing

1. Follow the coding standards in `GOOD_PRACTICES.md`
2. Ensure all tests pass: `make test`
3. Run code quality checks: `make lint format`
4. Submit pull requests with clear descriptions

## License

MIT License - see LICENSE file for details.
