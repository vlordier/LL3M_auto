# LL3M Local Development Setup Guide

This guide will help you set up LL3M for local development and testing with LM Studio (local LLM) and Blender on macOS.

## Prerequisites

### Required Software

1. **Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Blender 4.0+**
   - Download from: https://www.blender.org/download/
   - Install to `/Applications/Blender.app` (default location)
   - Verify installation:
     ```bash
     /Applications/Blender.app/Contents/MacOS/Blender --version
     ```

3. **LM Studio**
   - Download from: https://lmstudio.ai/
   - Install and launch LM Studio
   - Download and load a model (recommended: Llama 2 7B or similar)

4. **Git**
   ```bash
   git --version
   ```

### Recommended Models for LM Studio

For best results with LL3M, use these models in LM Studio:
- **Llama 2 7B Chat** (good balance of performance and speed)
- **Code Llama 7B** (optimized for code generation)
- **Mistral 7B Instruct** (fast and capable)
- **Llama 3 8B Instruct** (latest and most capable)

## Step-by-Step Setup

### 1. Clone and Install LL3M

```bash
# Clone the repository
git clone <repository-url>
cd LL3M_auto

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Setup

```bash
# Make setup script executable
chmod +x setup/blender_mcp_setup.py

# Run the setup script
python setup/blender_mcp_setup.py
```

The setup script will:
- ✅ Detect Blender installation
- ✅ Install required Python packages in Blender
- ✅ Create Blender MCP server script
- ✅ Generate startup and test scripts
- ✅ Create environment template

### 3. Configure Environment

```bash
# Copy the template environment file
cp .env.local .env

# Edit the configuration (optional)
nano .env
```

Key environment variables:
```env
# Use local LLM instead of OpenAI
USE_LOCAL_LLM=true

# LM Studio configuration
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=local-model

# Blender configuration
BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender
BLENDER_MCP_SERVER_URL=http://localhost:3001

# Development mode
DEVELOPMENT=true
DEBUG=true
```

### 4. Start LM Studio

1. Launch LM Studio
2. Go to the **Models** tab and download a model
3. Go to the **Chat** tab and load the model
4. Go to the **Developer** tab and start the server
5. Verify the server is running at `http://localhost:1234`

### 5. Start Blender MCP Server

```bash
# Start the Blender MCP server
./setup/start_blender_mcp.sh

# In a separate terminal, verify it's running
curl http://localhost:3001/health
```

You should see output like:
```json
{"status": "healthy", "blender_version": "4.0.0"}
```

### 6. Run Tests

```bash
# Test Blender MCP integration
./setup/test_blender_mcp.py

# Run unit tests
python -m pytest tests/unit/ -v

# Run E2E tests (requires LM Studio and Blender MCP running)
python -m pytest tests/e2e/ -v -s
```

## Usage Examples

### Basic Usage with Local LLM

```python
from src.utils.llm_client import get_llm_client
from src.utils.config import get_settings

# Configure for local development
settings = get_settings()
settings.app.use_local_llm = True

# Get LLM client (will use LM Studio)
client = get_llm_client()

# Generate Blender code
messages = [
    {"role": "user", "content": "Create a red cube in Blender"}
]

response = await client.chat_completion(messages)
print(response["choices"][0]["message"]["content"])
```

### Using Blender MCP Server

```python
import aiohttp

async def create_blender_scene():
    async with aiohttp.ClientSession() as session:
        # Execute Blender code
        code = """
        import bpy
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        print("Cube created!")
        """

        async with session.post(
            "http://localhost:3001/execute",
            json={"code": code}
        ) as response:
            result = await response.json()
            print(f"Success: {result['success']}")
            print(f"Logs: {result['logs']}")
```

### Running Full Workflow

```python
from src.workflow.graph import create_initial_workflow
from src.utils.types import WorkflowState

# Create workflow
workflow = create_initial_workflow()

# Run with a prompt
initial_state = WorkflowState(
    prompt="Create a blue sphere with a red cube next to it"
)

result = await workflow.ainvoke(initial_state)
print(f"Workflow completed: {result}")
```

## Troubleshooting

### Common Issues

#### LM Studio Connection Issues
```bash
# Check if LM Studio server is running
curl http://localhost:1234/v1/models

# If no response, ensure:
# 1. LM Studio is running
# 2. A model is loaded
# 3. Server is started in Developer tab
```

#### Blender MCP Server Issues
```bash
# Check Blender installation
/Applications/Blender.app/Contents/MacOS/Blender --version

# Check if MCP server is running
curl http://localhost:3001/health

# View server logs
./setup/start_blender_mcp.sh  # Look for error messages
```

#### Python Package Issues in Blender
```bash
# Manually install packages in Blender's Python
/Applications/Blender.app/Contents/MacOS/Blender --background --python-expr "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fastapi', 'uvicorn'])
"
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x setup/*.py
chmod +x setup/*.sh

# Check file permissions
ls -la setup/
```

### Performance Optimization

#### For LM Studio
- Use appropriate model size for your hardware (7B for 16GB RAM, 13B for 32GB+)
- Increase context length in LM Studio settings for complex prompts
- Enable GPU acceleration if available

#### For Blender
- Use headless mode for better performance: `BLENDER_HEADLESS=true`
- Increase timeout for complex scenes: `BLENDER_TIMEOUT=600`
- Monitor memory usage with complex scenes

## Testing

### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_asset_manager.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

### E2E Tests
```bash
# Ensure prerequisites are running:
# 1. LM Studio with model loaded
# 2. Blender MCP server

# Run all E2E tests
python -m pytest tests/e2e/ -v -s

# Run specific E2E test
python -m pytest tests/e2e/test_full_workflow.py::test_simple_workflow_e2e -v -s
```

### Test Configuration

E2E tests require:
- `USE_LOCAL_LLM=true` in environment
- LM Studio running with a model
- Blender MCP server running
- Development mode enabled

## Development Workflow

### Typical Development Session

1. **Start services**:
   ```bash
   # Terminal 1: Start LM Studio (GUI)
   # Terminal 2: Start Blender MCP
   ./setup/start_blender_mcp.sh
   ```

2. **Run tests**:
   ```bash
   # Terminal 3: Run tests
   python -m pytest tests/e2e/test_full_workflow.py -v -s
   ```

3. **Develop and iterate**:
   ```bash
   # Make changes to code
   # Run specific tests
   python -m pytest tests/e2e/test_simple_workflow_e2e -v -s
   ```

### Code Generation Tips

For best results with local LLMs:

1. **Use clear, specific prompts**:
   ```
   ❌ "Create a scene"
   ✅ "Create a Blender scene with a red cube at origin and blue sphere at (2,0,0)"
   ```

2. **Provide context**:
   ```python
   system_prompt = """You are a Blender Python expert. Generate clean code that:
   - Clears the scene first
   - Uses proper bpy operations
   - Includes error handling
   - Adds print statements for debugging"""
   ```

3. **Use lower temperature for code generation** (0.1-0.3)

4. **Handle code extraction**:
   ```python
   # Extract code from markdown blocks
   if "```python" in response:
       code = extract_code_block(response)
   ```

## Advanced Configuration

### Custom Model Configuration

```env
# Use specific model
LMSTUDIO_MODEL=llama-2-7b-chat.Q4_K_M.gguf

# Adjust generation parameters
LMSTUDIO_TEMPERATURE=0.2
LMSTUDIO_MAX_TOKENS=1500
LMSTUDIO_TIMEOUT=120
```

### Blender Customization

```env
# Custom Blender path
BLENDER_PATH=/usr/local/bin/blender

# Custom MCP server port
BLENDER_MCP_SERVER_PORT=3002
BLENDER_MCP_SERVER_URL=http://localhost:3002

# Larger timeout for complex scenes
BLENDER_TIMEOUT=600
```

### Development vs Production

Development mode (`.env`):
```env
DEVELOPMENT=true
DEBUG=true
LOG_LEVEL=DEBUG
USE_LOCAL_LLM=true
```

Production mode (`.env.production`):
```env
DEVELOPMENT=false
DEBUG=false
LOG_LEVEL=INFO
USE_LOCAL_LLM=false
OPENAI_API_KEY=your-production-key
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review test output for specific error messages
3. Check service logs (LM Studio console, Blender MCP output)
4. Verify environment configuration with `python -c "from src.utils.config import get_settings; print(get_settings().__dict__)"`

## Next Steps

Once setup is complete:
1. Run the example E2E tests to verify everything works
2. Explore the codebase structure in `src/`
3. Try creating custom prompts and scenes
4. Experiment with different LLM models in LM Studio
5. Contribute improvements and report issues
