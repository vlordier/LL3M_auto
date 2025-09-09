# LL3M Web Interface

A modern, ChatGPT-inspired web interface for the LL3M 3D asset generation system.

## Features

- 🤖 **ChatGPT-like Interface** - Familiar chat experience for asset generation
- 🔮 **Real-time 3D Preview** - Interactive 3D viewer with live updates
- 📋 **Visual Workflow Steps** - See generation progress with step-by-step previews  
- 🔄 **Session Management** - Resume, branch, and amend previous generations
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- ⚡ **Real-time Updates** - WebSocket-powered live progress tracking

## Architecture

```
web/
├── backend/           # FastAPI server with WebSocket support
│   ├── api/          # REST API endpoints  
│   ├── websocket/    # Real-time communication
│   └── services/     # Business logic integration
└── frontend/         # React TypeScript application
    ├── src/
    │   ├── components/
    │   │   ├── Chat/           # ChatGPT-like interface
    │   │   ├── Viewer3D/       # Three.js 3D preview
    │   │   ├── WorkflowSteps/  # Step visualization
    │   │   └── SessionBrowser/ # Session management
    │   ├── hooks/              # Custom React hooks
    │   ├── stores/             # Zustand state management
    │   └── types/              # TypeScript definitions
    └── public/                 # Static assets
```

## Technology Stack

**Frontend:**
- React 18 + TypeScript
- Tailwind CSS for styling  
- Three.js + React Three Fiber for 3D rendering
- @react-three/drei for 3D helpers
- Framer Motion for animations
- Zustand for state management
- Socket.IO client for real-time updates

**Backend:**
- FastAPI for async Python API
- WebSocket support for real-time communication  
- Integration with existing LL3M session system
- File serving for 3D assets and previews

## User Flows

### 1. New Asset Generation
```
User enters prompt → Real-time generation steps → Interactive 3D preview
```

### 2. Session Resumption  
```
Browse sessions → Select session → Continue from any step → Provide amendments
```

### 3. Workflow Amendment
```
Click on any step → Provide feedback → System generates alternative path
```

## Getting Started

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Install frontend dependencies  
cd ../frontend
npm install

# Start development server
npm run dev
```

Navigate to `http://localhost:3000` to access the web interface.

## Backend Modes

- Full mode (recommended): runs with the main LL3M backend (uses project dependencies and Python 3.12+).
  - Commands:
    - Start backend: `make web-backend`
    - Start full stack: `make web-run`

- Standalone mode (lightweight): runs only web/backend with its own requirements; generation/session features are disabled.
  - Commands:
    - Install deps: `make web-backend-setup`
    - Start backend: `make web-backend-standalone`
    - Start full stack with standalone: `make web-run-standalone`

Notes:
- Standalone mode is useful for UI development and e2e tests that mock API calls. Endpoints that require generation or session orchestration return HTTP 503 in this mode.
- Full mode provides all features. Ensure Python 3.12+ and run `pip install -e .` (or `make install`).

## OpenAI Integration & E2E Tests

To run the web with OpenAI integration and execute the Playwright e2e tests:

1) Configure environment

```bash
# At repo root
make web-config     # creates .env if missing

# Provide your OpenAI API key (either export or add to .env)
export OPENAI_API_KEY=sk-...
# or edit .env and set OPENAI_API_KEY

# Optional: force OpenAI instead of local LLM
export USE_LOCAL_LLM=false
```

2) Start services (OpenAI mode)

```bash
# Start both backend and frontend with OpenAI integration
make web-run-openai
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

3) Run Playwright e2e tests

```bash
# From repo root (spawns tests in web/frontend)
make web-test        # headless
# or
make web-test-headed # with browser UI
```

Shortcut demo (start OpenAI mode in background, wait, then run headed tests):

```bash
make web-demo
```

Artifacts:
- Reports: `web/frontend/playwright-report/`
- Videos: `web/frontend/test-results/`
