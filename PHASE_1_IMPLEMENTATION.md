# Phase 1: Foundation & Setup - Implementation Details

## Overview
Phase 1 establishes the foundational infrastructure for LL3M, including project structure, dependencies, Context7 MCP integration, and basic Blender execution capabilities.

**Duration**: 1-2 weeks  
**Branch**: `phase-1/foundation-setup`

## Success Criteria
- [ ] Project structure established with proper organization
- [ ] All dependencies installed and configured
- [ ] Context7 MCP integration working for Blender documentation
- [ ] Basic Blender execution capability functional
- [ ] Development environment fully configured
- [ ] Basic tests passing

---

## Task 1: Project Structure Setup

### 1.1 Create Directory Structure
```bash
mkdir -p src/{agents,knowledge,blender,workflow,utils}
mkdir -p config tests examples docs
mkdir -p outputs logs
```

**Expected Structure:**
```
ll3m/
├── src/
│   ├── __init__.py
│   ├── agents/              # LangGraph agent implementations
│   │   ├── __init__.py
│   │   ├── base.py         # Base agent class
│   │   ├── planner.py      # Planner agent
│   │   ├── retrieval.py    # Retrieval agent  
│   │   ├── coding.py       # Coding agent
│   │   ├── critic.py       # Critic agent
│   │   └── verification.py # Verification agent
│   ├── knowledge/          # Context7 MCP integration
│   │   ├── __init__.py
│   │   ├── context7_client.py
│   │   └── retrieval_service.py
│   ├── blender/            # Blender API wrappers
│   │   ├── __init__.py
│   │   ├── executor.py     # Blender execution engine
│   │   ├── templates.py    # Code templates
│   │   └── utils.py        # Blender utilities
│   ├── workflow/           # LangGraph workflow definitions
│   │   ├── __init__.py
│   │   ├── state.py        # State definitions
│   │   ├── nodes.py        # Workflow nodes
│   │   └── graph.py        # Main workflow graph
│   └── utils/              # Utilities and helpers
│       ├── __init__.py
│       ├── logging.py      # Logging configuration
│       ├── config.py       # Configuration management
│       └── types.py        # Type definitions
├── config/                 # Configuration files
│   ├── agents.yaml        # Agent configurations
│   ├── blender.yaml       # Blender settings
│   └── development.env    # Environment variables
├── tests/                 # Test suites
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_knowledge/
│   ├── test_blender/
│   └── test_workflow/
├── examples/              # Example prompts and outputs
│   ├── prompts/
│   └── outputs/
├── docs/                  # Documentation
├── outputs/               # Generated assets
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── .gitignore
├── .env.example
├── README.md
└── Makefile              # Development commands
```

### 1.2 Initialize Python Package
```python
# src/__init__.py
"""LL3M: Large Language 3D Modelers"""

__version__ = "0.1.0"
__author__ = "LL3M Team"
__description__ = "Multi-agent system for generating 3D assets using LLMs"
```

---

## Task 2: Dependencies Configuration

### 2.1 Create requirements.txt
```txt
# Core dependencies
langgraph>=0.0.40
openai>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Async support
asyncio
aiohttp>=3.8.0
aiofiles>=22.0.0

# Configuration
python-dotenv>=1.0.0
PyYAML>=6.0
click>=8.0.0

# Logging and monitoring
structlog>=22.0.0
rich>=13.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0
pytest-cov>=4.0.0

# Development
black>=22.0.0
isort>=5.12.0
mypy>=1.0.0
pre-commit>=3.0.0

# Blender integration (will be installed separately)
# Note: bpy (Blender Python API) requires special installation
```

### 2.2 Create pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ll3m"
version = "0.1.0"
description = "Large Language 3D Modelers - Multi-agent system for generating 3D assets"
authors = [{name = "LL3M Team"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "langgraph>=0.0.40",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "aiohttp>=3.8.0",
    "aiofiles>=22.0.0",
    "python-dotenv>=1.0.0",
    "PyYAML>=6.0",
    "click>=8.0.0",
    "structlog>=22.0.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.10.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
ll3m = "src.cli:main"

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
```

### 2.3 Environment Configuration
```env
# .env.example
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000

# Context7 MCP Configuration
CONTEXT7_MCP_SERVER=context7_server_url
CONTEXT7_API_KEY=your_context7_key

# Blender Configuration
BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender
BLENDER_HEADLESS=true
BLENDER_TIMEOUT=300

# Application Configuration
LOG_LEVEL=INFO
OUTPUT_DIRECTORY=./outputs
MAX_REFINEMENT_ITERATIONS=3
ENABLE_ASYNC=true

# Development Configuration
DEVELOPMENT=true
DEBUG=false
```

---

## Task 3: Core Type Definitions

### 3.1 Create src/utils/types.py
```python
"""Type definitions for LL3M system."""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TaskType(str, Enum):
    """Types of 3D modeling subtasks."""
    GEOMETRY = "geometry"
    MATERIAL = "material"
    LIGHTING = "lighting"
    ANIMATION = "animation"
    SCENE_SETUP = "scene_setup"


class IssueType(str, Enum):
    """Types of issues that can be detected in 3D assets."""
    GEOMETRY_ERROR = "geometry_error"
    MATERIAL_ISSUE = "material_issue"
    LIGHTING_PROBLEM = "lighting_problem"
    SCALE_ISSUE = "scale_issue"
    POSITIONING_ERROR = "positioning_error"


class AgentType(str, Enum):
    """Types of agents in the system."""
    PLANNER = "planner"
    RETRIEVAL = "retrieval"
    CODING = "coding"
    CRITIC = "critic"
    VERIFICATION = "verification"


class SubTask(BaseModel):
    """A subtask identified by the planner."""
    id: str = Field(..., description="Unique identifier for the subtask")
    type: TaskType = Field(..., description="Type of the subtask")
    description: str = Field(..., description="Description of what needs to be done")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    dependencies: List[str] = Field(default=[], description="IDs of dependent subtasks")
    parameters: Dict[str, Any] = Field(default={}, description="Task-specific parameters")


class Issue(BaseModel):
    """An issue detected in a 3D asset."""
    id: str = Field(..., description="Unique identifier for the issue")
    type: IssueType = Field(..., description="Type of the issue")
    description: str = Field(..., description="Description of the issue")
    severity: int = Field(ge=1, le=5, description="Severity level (1-5)")
    suggested_fix: str = Field(..., description="Suggested fix for the issue")
    code_location: Optional[str] = Field(None, description="Location in code that needs fixing")


class ExecutionResult(BaseModel):
    """Result of executing Blender code."""
    success: bool = Field(..., description="Whether execution was successful")
    asset_path: Optional[str] = Field(None, description="Path to generated asset file")
    screenshot_path: Optional[str] = Field(None, description="Path to screenshot")
    logs: List[str] = Field(default=[], description="Execution logs")
    errors: List[str] = Field(default=[], description="Execution errors")
    execution_time: float = Field(..., description="Execution time in seconds")


class AssetMetadata(BaseModel):
    """Metadata for a generated 3D asset."""
    id: str = Field(..., description="Unique identifier for the asset")
    prompt: str = Field(..., description="Original text prompt")
    creation_time: datetime = Field(default_factory=datetime.now)
    file_path: str = Field(..., description="Path to the asset file")
    screenshot_path: Optional[str] = Field(None, description="Path to screenshot")
    subtasks: List[SubTask] = Field(default=[], description="Subtasks used to create asset")
    refinement_count: int = Field(default=0, description="Number of refinements applied")
    quality_score: Optional[float] = Field(None, ge=0, le=1, description="Quality assessment score")


class AgentResponse(BaseModel):
    """Standard response format for agents."""
    agent_type: AgentType = Field(..., description="Type of agent that generated response")
    success: bool = Field(..., description="Whether the operation was successful")
    data: Any = Field(None, description="Response data")
    message: str = Field("", description="Human-readable message")
    execution_time: float = Field(..., description="Time taken to process")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class WorkflowState(BaseModel):
    """State object for LangGraph workflow."""
    # Input
    prompt: str = Field(..., description="Original user prompt")
    user_feedback: Optional[str] = Field(None, description="User refinement feedback")
    
    # Planning phase
    subtasks: List[SubTask] = Field(default=[], description="Identified subtasks")
    
    # Retrieval phase
    documentation: str = Field("", description="Retrieved Blender documentation")
    
    # Coding phase
    generated_code: str = Field("", description="Generated Blender Python code")
    
    # Execution phase
    execution_result: Optional[ExecutionResult] = Field(None, description="Execution result")
    
    # Analysis phase
    detected_issues: List[Issue] = Field(default=[], description="Issues detected by critic")
    
    # Refinement tracking
    refinement_count: int = Field(default=0, description="Number of refinement iterations")
    max_refinements: int = Field(default=3, description="Maximum allowed refinements")
    
    # Asset tracking
    asset_metadata: Optional[AssetMetadata] = Field(None, description="Generated asset metadata")
    
    # Workflow control
    should_continue: bool = Field(default=True, description="Whether to continue refinement")
    error_message: Optional[str] = Field(None, description="Error message if workflow failed")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
```

---

## Task 4: Configuration Management

### 4.1 Create src/utils/config.py
```python
"""Configuration management for LL3M."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Default model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    
    model_config = SettingsConfigDict(env_prefix="OPENAI_")


class Context7Config(BaseSettings):
    """Context7 MCP configuration."""
    
    mcp_server: str = Field(..., description="Context7 MCP server URL")
    api_key: Optional[str] = Field(None, description="Context7 API key if required")
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")
    
    model_config = SettingsConfigDict(env_prefix="CONTEXT7_")


class BlenderConfig(BaseSettings):
    """Blender execution configuration."""
    
    path: str = Field(..., description="Path to Blender executable")
    headless: bool = Field(default=True, description="Run Blender in headless mode")
    timeout: int = Field(default=300, gt=0, description="Execution timeout in seconds")
    screenshot_resolution: tuple[int, int] = Field(default=(800, 600))
    
    model_config = SettingsConfigDict(env_prefix="BLENDER_")


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    log_level: str = Field(default="INFO", description="Logging level")
    output_directory: Path = Field(default=Path("./outputs"), description="Output directory")
    max_refinement_iterations: int = Field(default=3, ge=1, description="Max refinements")
    enable_async: bool = Field(default=True, description="Enable async processing")
    development: bool = Field(default=False, description="Development mode")
    debug: bool = Field(default=False, description="Debug mode")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


class Settings:
    """Global settings manager."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize settings."""
        if env_file and os.path.exists(env_file):
            # Load from specific env file
            self.app = AppConfig(_env_file=env_file)
        else:
            # Load from default .env or environment variables
            self.app = AppConfig()
            
        self.openai = OpenAIConfig()
        self.context7 = Context7Config()
        self.blender = BlenderConfig()
        
        # Ensure output directory exists
        self.app.output_directory.mkdir(parents=True, exist_ok=True)
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for specific agent type."""
        # This will be loaded from YAML files later
        default_configs = {
            "planner": {
                "model": self.openai.model,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "retrieval": {
                "model": self.openai.model,
                "temperature": 0.3,
                "max_tokens": 1500,
            },
            "coding": {
                "model": self.openai.model,
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            "critic": {
                "model": "gpt-4-vision-preview",
                "temperature": 0.5,
                "max_tokens": 1500,
            },
            "verification": {
                "model": self.openai.model,
                "temperature": 0.3,
                "max_tokens": 1000,
            }
        }
        
        return default_configs.get(agent_type, {
            "model": self.openai.model,
            "temperature": self.openai.temperature,
            "max_tokens": self.openai.max_tokens,
        })


# Global settings instance
settings = Settings()
```

---

## Task 5: Context7 MCP Integration

### 5.1 Create src/knowledge/context7_client.py
```python
"""Context7 MCP client for Blender documentation retrieval."""

import asyncio
from typing import List, Optional, Dict, Any
import aiohttp
import structlog

from ..utils.config import settings
from ..utils.types import AgentResponse, AgentType

logger = structlog.get_logger(__name__)


class Context7MCPClient:
    """Client for interacting with Context7 MCP server."""
    
    def __init__(self):
        """Initialize the Context7 MCP client."""
        self.server_url = settings.context7.mcp_server
        self.api_key = settings.context7.api_key
        self.timeout = settings.context7.timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def resolve_library_id(self, library_name: str) -> Optional[str]:
        """Resolve a package name to Context7-compatible library ID."""
        try:
            # For Blender, we'll search for 'blender' or 'bpy'
            if 'blender' in library_name.lower() or library_name.lower() == 'bpy':
                # This would be the actual MCP call - placeholder implementation
                return '/blender/python-api'
            
            logger.info("Resolving library ID", library_name=library_name)
            
            # Placeholder: In real implementation, this would call the MCP server
            # For now, return a default Blender library ID
            return '/blender/python-api'
            
        except Exception as e:
            logger.error("Failed to resolve library ID", error=str(e), library_name=library_name)
            return None
    
    async def get_library_docs(
        self, 
        library_id: str, 
        topic: Optional[str] = None,
        tokens: int = 10000
    ) -> Optional[str]:
        """Fetch documentation for a specific library."""
        try:
            logger.info(
                "Fetching library documentation", 
                library_id=library_id, 
                topic=topic, 
                tokens=tokens
            )
            
            # Placeholder: In real implementation, this would call the MCP server
            # For now, return sample Blender documentation
            return self._get_sample_blender_docs(topic)
            
        except Exception as e:
            logger.error("Failed to fetch library docs", error=str(e), library_id=library_id)
            return None
    
    def _get_sample_blender_docs(self, topic: Optional[str] = None) -> str:
        """Return sample Blender documentation for testing."""
        base_docs = """
# Blender Python API Documentation

## Basic Operations

### Creating Objects
```python
import bpy

# Add a cube
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Add a sphere
bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))

# Add a cylinder
bpy.ops.mesh.primitive_cylinder_add(location=(4, 0, 0))
```

### Materials
```python
# Create a new material
material = bpy.data.materials.new(name="MyMaterial")
material.use_nodes = True

# Get the material's node tree
nodes = material.node_tree.nodes
bsdf = nodes.get("Principled BSDF")

# Set base color
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)  # Red
```

### Scene Management
```python
# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Set viewport shading
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].shading.type = 'MATERIAL'
```
"""
        
        if topic:
            topic_docs = {
                "geometry": """
### Geometry Operations
```python
# Modify object geometry
bpy.context.object.data.vertices[0].co = (1, 1, 1)

# Apply modifiers
modifier = bpy.context.object.modifiers.new(name="Subsurf", type='SUBSURF')
modifier.levels = 2
```
""",
                "material": """
### Advanced Materials
```python
# Create node-based material
mat = bpy.data.materials.new(name="NodeMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes

# Add texture node
tex_node = nodes.new('ShaderNodeTexImage')
```
""",
                "lighting": """
### Lighting Setup
```python
# Add light
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.object
light.data.energy = 3.0
```
"""
            }
            
            return base_docs + topic_docs.get(topic, "")
        
        return base_docs


class Context7RetrievalService:
    """Service for retrieving Blender documentation using Context7."""
    
    def __init__(self):
        """Initialize the retrieval service."""
        self.client = Context7MCPClient()
    
    async def retrieve_documentation(
        self, 
        subtasks: List[str], 
        context: Optional[str] = None
    ) -> AgentResponse:
        """Retrieve relevant Blender documentation for given subtasks."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.client as client:
                # Resolve Blender library ID
                library_id = await client.resolve_library_id("blender")
                if not library_id:
                    return AgentResponse(
                        agent_type=AgentType.RETRIEVAL,
                        success=False,
                        message="Failed to resolve Blender library ID",
                        execution_time=asyncio.get_event_loop().time() - start_time
                    )
                
                # Determine topic from subtasks
                topic = self._extract_topic_from_subtasks(subtasks)
                
                # Fetch documentation
                docs = await client.get_library_docs(library_id, topic=topic)
                if not docs:
                    return AgentResponse(
                        agent_type=AgentType.RETRIEVAL,
                        success=False,
                        message="Failed to fetch Blender documentation",
                        execution_time=asyncio.get_event_loop().time() - start_time
                    )
                
                logger.info(
                    "Successfully retrieved documentation", 
                    topic=topic, 
                    doc_length=len(docs)
                )
                
                return AgentResponse(
                    agent_type=AgentType.RETRIEVAL,
                    success=True,
                    data=docs,
                    message=f"Retrieved Blender documentation for topic: {topic}",
                    execution_time=asyncio.get_event_loop().time() - start_time,
                    metadata={"topic": topic, "library_id": library_id}
                )
                
        except Exception as e:
            logger.error("Documentation retrieval failed", error=str(e))
            return AgentResponse(
                agent_type=AgentType.RETRIEVAL,
                success=False,
                message=f"Documentation retrieval failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    def _extract_topic_from_subtasks(self, subtasks: List[str]) -> Optional[str]:
        """Extract the primary topic from subtasks."""
        # Simple keyword matching - can be improved with ML
        subtasks_text = " ".join(subtasks).lower()
        
        if any(keyword in subtasks_text for keyword in ['geometry', 'mesh', 'vertex', 'face']):
            return 'geometry'
        elif any(keyword in subtasks_text for keyword in ['material', 'shader', 'texture']):
            return 'material'
        elif any(keyword in subtasks_text for keyword in ['light', 'illumination', 'shadow']):
            return 'lighting'
        
        return None
```

---

## Task 6: Basic Blender Integration

### 6.1 Create src/blender/executor.py
```python
"""Blender execution engine for running generated Python code."""

import asyncio
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import structlog

from ..utils.config import settings
from ..utils.types import ExecutionResult

logger = structlog.get_logger(__name__)


class BlenderExecutor:
    """Executes Python code in Blender environment."""
    
    def __init__(self):
        """Initialize the Blender executor."""
        self.blender_path = settings.blender.path
        self.headless = settings.blender.headless
        self.timeout = settings.blender.timeout
        self.output_dir = settings.app.output_directory
        
        # Ensure Blender is available
        self._validate_blender_installation()
    
    def _validate_blender_installation(self) -> None:
        """Validate that Blender is installed and accessible."""
        if not Path(self.blender_path).exists():
            raise RuntimeError(f"Blender not found at: {self.blender_path}")
        
        logger.info("Blender installation validated", path=self.blender_path)
    
    async def execute_code(
        self, 
        code: str, 
        asset_name: str = "asset",
        export_format: str = "blend"
    ) -> ExecutionResult:
        """Execute Blender Python code and return result."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_content = self._wrap_code_for_execution(code, asset_name, export_format)
                script_file.write(script_content)
                script_path = script_file.name
            
            # Execute Blender with script
            result = await self._run_blender_script(script_path)
            
            # Clean up temporary file
            Path(script_path).unlink(missing_ok=True)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            if result.success:
                logger.info(
                    "Blender execution successful", 
                    asset_name=asset_name,
                    execution_time=execution_time
                )
            else:
                logger.error(
                    "Blender execution failed", 
                    asset_name=asset_name,
                    errors=result.errors
                )
            
            return ExecutionResult(
                success=result.success,
                asset_path=result.asset_path,
                screenshot_path=result.screenshot_path,
                logs=result.logs,
                errors=result.errors,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error("Blender execution exception", error=str(e))
            
            return ExecutionResult(
                success=False,
                logs=[],
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _wrap_code_for_execution(
        self, 
        code: str, 
        asset_name: str,
        export_format: str
    ) -> str:
        """Wrap user code with necessary Blender setup and export logic."""
        output_dir = self.output_dir
        asset_path = output_dir / f"{asset_name}.{export_format}"
        screenshot_path = output_dir / f"{asset_name}_screenshot.png"
        
        wrapped_code = f'''
import bpy
import bmesh
import sys
import traceback
import json
from pathlib import Path

# Setup
output_dir = Path("{output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

asset_path = output_dir / "{asset_name}.{export_format}"
screenshot_path = output_dir / "{asset_name}_screenshot.png"

logs = []
errors = []
success = False

try:
    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    
    # Execute user code
    logs.append("Starting user code execution")
    
{self._indent_code(code, "    ")}
    
    logs.append("User code executed successfully")
    
    # Save the file
    bpy.ops.wm.save_as_mainfile(filepath=str(asset_path))
    logs.append(f"Asset saved to: {{asset_path}}")
    
    # Take screenshot
    bpy.context.scene.render.filepath = str(screenshot_path)
    bpy.context.scene.render.resolution_x = {settings.blender.screenshot_resolution[0]}
    bpy.context.scene.render.resolution_y = {settings.blender.screenshot_resolution[1]}
    bpy.ops.render.render(write_still=True)
    logs.append(f"Screenshot saved to: {{screenshot_path}}")
    
    success = True
    
except Exception as e:
    error_msg = f"Execution error: {{str(e)}}"
    errors.append(error_msg)
    errors.append(traceback.format_exc())
    logs.append(f"Error occurred: {{error_msg}}")

# Output result as JSON for parsing
result = {{
    "success": success,
    "asset_path": str(asset_path) if success else None,
    "screenshot_path": str(screenshot_path) if success else None,
    "logs": logs,
    "errors": errors
}}

print("EXECUTION_RESULT_JSON:", json.dumps(result))
'''
        return wrapped_code
    
    def _indent_code(self, code: str, indent: str) -> str:
        """Indent code lines."""
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)
    
    async def _run_blender_script(self, script_path: str) -> ExecutionResult:
        """Run Blender with the given script."""
        cmd = [
            self.blender_path,
            "--background",
            "--python", script_path
        ]
        
        if self.headless:
            cmd.insert(1, "--no-window")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout
            )
            
            stdout_text = stdout.decode('utf-8')
            stderr_text = stderr.decode('utf-8')
            
            # Parse result from stdout
            result = self._parse_execution_result(stdout_text, stderr_text)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Blender execution timed out", timeout=self.timeout)
            return ExecutionResult(
                success=False,
                logs=[],
                errors=[f"Execution timed out after {self.timeout} seconds"],
                execution_time=float(self.timeout)
            )
        except Exception as e:
            logger.error("Failed to run Blender", error=str(e))
            return ExecutionResult(
                success=False,
                logs=[],
                errors=[f"Failed to run Blender: {str(e)}"],
                execution_time=0.0
            )
    
    def _parse_execution_result(self, stdout: str, stderr: str) -> ExecutionResult:
        """Parse execution result from Blender output."""
        try:
            # Look for our JSON result marker
            for line in stdout.split('\n'):
                if line.startswith("EXECUTION_RESULT_JSON:"):
                    json_str = line.replace("EXECUTION_RESULT_JSON:", "").strip()
                    result_data = json.loads(json_str)
                    
                    return ExecutionResult(
                        success=result_data["success"],
                        asset_path=result_data.get("asset_path"),
                        screenshot_path=result_data.get("screenshot_path"),
                        logs=result_data.get("logs", []),
                        errors=result_data.get("errors", []),
                        execution_time=0.0  # Will be set by caller
                    )
        except Exception as e:
            logger.error("Failed to parse execution result", error=str(e))
        
        # Fallback: parse from stderr/stdout
        return ExecutionResult(
            success="Error" not in stderr and process.returncode == 0 if 'process' in locals() else False,
            logs=stdout.split('\n') if stdout else [],
            errors=stderr.split('\n') if stderr else [],
            execution_time=0.0
        )
    
    async def take_screenshot(self, blend_file_path: str, output_path: str) -> bool:
        """Take a screenshot of a Blender file."""
        try:
            script_content = f'''
import bpy

# Open the blend file
bpy.ops.wm.open_mainfile(filepath="{blend_file_path}")

# Set up rendering
bpy.context.scene.render.filepath = "{output_path}"
bpy.context.scene.render.resolution_x = {settings.blender.screenshot_resolution[0]}
bpy.context.scene.render.resolution_y = {settings.blender.screenshot_resolution[1]}

# Render screenshot
bpy.ops.render.render(write_still=True)
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(script_content)
                script_path = script_file.name
            
            result = await self._run_blender_script(script_path)
            Path(script_path).unlink(missing_ok=True)
            
            return result.success
            
        except Exception as e:
            logger.error("Failed to take screenshot", error=str(e))
            return False
```

---

## Task 7: Logging Configuration

### 7.1 Create src/utils/logging.py
```python
"""Logging configuration for LL3M."""

import sys
import structlog
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def setup_logging() -> None:
    """Set up structured logging with Rich formatting."""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.app.development else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, settings.app.log_level.upper(), structlog.INFO)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set up rich console for pretty output
    console = Console()
    
    # Configure rich handler for development
    if settings.app.development:
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True
        )
        
        # Add rich handler to root logger for pretty development output
        import logging
        logging.basicConfig(
            level=getattr(logging, settings.app.log_level.upper()),
            format="%(message)s",
            handlers=[rich_handler]
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger."""
    return structlog.get_logger(name)
```

---

## Task 8: Basic Tests

### 8.1 Create tests/test_config.py
```python
"""Test configuration management."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.config import Settings, AppConfig, OpenAIConfig


class TestSettings:
    """Test settings management."""
    
    def test_app_config_defaults(self):
        """Test default app configuration."""
        config = AppConfig()
        
        assert config.log_level == "INFO"
        assert config.output_directory == Path("./outputs")
        assert config.max_refinement_iterations == 3
        assert config.enable_async is True
        assert config.development is False
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'OPENAI_MODEL': 'gpt-4'})
    def test_openai_config_from_env(self):
        """Test OpenAI config from environment variables."""
        config = OpenAIConfig()
        
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
    
    def test_settings_initialization(self):
        """Test settings initialization."""
        settings = Settings()
        
        assert settings.app is not None
        assert settings.openai is not None
        assert settings.context7 is not None
        assert settings.blender is not None
    
    def test_get_agent_config(self):
        """Test agent configuration retrieval."""
        settings = Settings()
        
        planner_config = settings.get_agent_config("planner")
        assert "model" in planner_config
        assert "temperature" in planner_config
        assert "max_tokens" in planner_config
        
        # Test unknown agent type
        unknown_config = settings.get_agent_config("unknown")
        assert "model" in unknown_config


class TestTypes:
    """Test type definitions."""
    
    def test_subtask_creation(self):
        """Test SubTask model creation."""
        from src.utils.types import SubTask, TaskType
        
        subtask = SubTask(
            id="test-1",
            type=TaskType.GEOMETRY,
            description="Create a cube"
        )
        
        assert subtask.id == "test-1"
        assert subtask.type == TaskType.GEOMETRY
        assert subtask.description == "Create a cube"
        assert subtask.priority == 1
        assert subtask.dependencies == []
    
    def test_execution_result_creation(self):
        """Test ExecutionResult model creation."""
        from src.utils.types import ExecutionResult
        
        result = ExecutionResult(
            success=True,
            asset_path="/path/to/asset.blend",
            execution_time=2.5
        )
        
        assert result.success is True
        assert result.asset_path == "/path/to/asset.blend"
        assert result.execution_time == 2.5
        assert result.logs == []
        assert result.errors == []
```

### 8.2 Create tests/test_context7.py
```python
"""Test Context7 MCP integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.knowledge.context7_client import Context7MCPClient, Context7RetrievalService
from src.utils.types import AgentType


class TestContext7MCPClient:
    """Test Context7 MCP client."""
    
    @pytest.mark.asyncio
    async def test_resolve_library_id(self):
        """Test library ID resolution."""
        client = Context7MCPClient()
        
        # Test Blender library resolution
        library_id = await client.resolve_library_id("blender")
        assert library_id == '/blender/python-api'
        
        library_id = await client.resolve_library_id("bpy")
        assert library_id == '/blender/python-api'
    
    @pytest.mark.asyncio
    async def test_get_library_docs(self):
        """Test documentation retrieval."""
        client = Context7MCPClient()
        
        docs = await client.get_library_docs('/blender/python-api')
        assert docs is not None
        assert 'Blender Python API' in docs
        assert 'bpy.ops.mesh.primitive_cube_add' in docs
    
    @pytest.mark.asyncio
    async def test_get_library_docs_with_topic(self):
        """Test documentation retrieval with specific topic."""
        client = Context7MCPClient()
        
        docs = await client.get_library_docs('/blender/python-api', topic='geometry')
        assert docs is not None
        assert 'Geometry Operations' in docs


class TestContext7RetrievalService:
    """Test Context7 retrieval service."""
    
    @pytest.mark.asyncio
    async def test_retrieve_documentation_success(self):
        """Test successful documentation retrieval."""
        service = Context7RetrievalService()
        
        subtasks = ["Create a cube", "Add materials"]
        response = await service.retrieve_documentation(subtasks)
        
        assert response.agent_type == AgentType.RETRIEVAL
        assert response.success is True
        assert response.data is not None
        assert 'Blender Python API' in response.data
    
    @pytest.mark.asyncio
    async def test_extract_topic_from_subtasks(self):
        """Test topic extraction from subtasks."""
        service = Context7RetrievalService()
        
        # Test geometry detection
        geometry_tasks = ["Create mesh geometry", "Modify vertices"]
        topic = service._extract_topic_from_subtasks(geometry_tasks)
        assert topic == 'geometry'
        
        # Test material detection
        material_tasks = ["Apply materials", "Set texture properties"]
        topic = service._extract_topic_from_subtasks(material_tasks)
        assert topic == 'material'
        
        # Test lighting detection
        lighting_tasks = ["Add sun light", "Configure illumination"]
        topic = service._extract_topic_from_subtasks(lighting_tasks)
        assert topic == 'lighting'
        
        # Test no specific topic
        general_tasks = ["Do something general"]
        topic = service._extract_topic_from_subtasks(general_tasks)
        assert topic is None
```

### 8.3 Create tests/test_blender_executor.py
```python
"""Test Blender executor."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from src.blender.executor import BlenderExecutor
from src.utils.types import ExecutionResult


class TestBlenderExecutor:
    """Test Blender execution."""
    
    def test_init_with_valid_blender(self):
        """Test executor initialization with valid Blender path."""
        with patch('pathlib.Path.exists', return_value=True):
            executor = BlenderExecutor()
            assert executor.blender_path is not None
    
    def test_init_with_invalid_blender(self):
        """Test executor initialization with invalid Blender path."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(RuntimeError, match="Blender not found"):
                BlenderExecutor()
    
    def test_wrap_code_for_execution(self):
        """Test code wrapping for execution."""
        with patch('pathlib.Path.exists', return_value=True):
            executor = BlenderExecutor()
            
            user_code = "bpy.ops.mesh.primitive_cube_add()"
            wrapped = executor._wrap_code_for_execution(user_code, "test_asset", "blend")
            
            assert "import bpy" in wrapped
            assert user_code in wrapped
            assert "test_asset.blend" in wrapped
            assert "EXECUTION_RESULT_JSON" in wrapped
    
    def test_indent_code(self):
        """Test code indentation."""
        with patch('pathlib.Path.exists', return_value=True):
            executor = BlenderExecutor()
            
            code = "line1\nline2\n  indented"
            indented = executor._indent_code(code, "    ")
            
            expected = "    line1\n    line2\n      indented"
            assert indented == expected
    
    def test_parse_execution_result_success(self):
        """Test parsing successful execution result."""
        with patch('pathlib.Path.exists', return_value=True):
            executor = BlenderExecutor()
            
            stdout = '''
Some Blender output
EXECUTION_RESULT_JSON: {"success": true, "asset_path": "/path/asset.blend", "logs": ["test"], "errors": []}
More output
'''
            stderr = ""
            
            result = executor._parse_execution_result(stdout, stderr)
            
            assert result.success is True
            assert result.asset_path == "/path/asset.blend"
            assert result.logs == ["test"]
            assert result.errors == []
    
    def test_parse_execution_result_failure(self):
        """Test parsing failed execution result."""
        with patch('pathlib.Path.exists', return_value=True):
            executor = BlenderExecutor()
            
            stdout = '''
EXECUTION_RESULT_JSON: {"success": false, "asset_path": null, "logs": [], "errors": ["Error occurred"]}
'''
            stderr = ""
            
            result = executor._parse_execution_result(stdout, stderr)
            
            assert result.success is False
            assert result.asset_path is None
            assert result.errors == ["Error occurred"]
```

---

## Task 9: Development Tools

### 9.1 Create .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
outputs/
logs/
*.blend
*.blend1
*.log

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/

# Jupyter
.ipynb_checkpoints

# Environments
.env.local
.env.development
.env.test
.env.production
```

### 9.2 Create Makefile
```makefile
.PHONY: install dev test clean lint format setup-blender help

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev          - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (mypy, flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  clean        - Clean up generated files"
	@echo "  setup-blender- Setup Blender for development"

# Installation
install:
	pip install -e .

dev: install
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	mypy src/
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

# Blender setup (platform specific)
setup-blender:
	@echo "Setting up Blender..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "macOS detected"; \
		if [ ! -f "/Applications/Blender.app/Contents/MacOS/Blender" ]; then \
			echo "Blender not found. Please install Blender from https://www.blender.org/download/"; \
		else \
			echo "Blender found at /Applications/Blender.app/Contents/MacOS/Blender"; \
		fi \
	elif [ "$(shell uname)" = "Linux" ]; then \
		echo "Linux detected"; \
		if ! command -v blender &> /dev/null; then \
			echo "Installing Blender..."; \
			sudo apt-get update && sudo apt-get install -y blender; \
		else \
			echo "Blender is already installed"; \
		fi \
	else \
		echo "Windows detected - please install Blender manually"; \
	fi

# Run the application
run-example:
	python -m src.main "Create a red cube with a metallic material"

# Development server (if we add FastAPI later)
dev-server:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Success Criteria Checklist

By the end of Phase 1, we should have:

### ✅ Project Structure
- [ ] Complete directory structure established
- [ ] All necessary `__init__.py` files created
- [ ] Package structure properly configured

### ✅ Dependencies & Configuration
- [ ] `requirements.txt` and `pyproject.toml` configured
- [ ] Environment variable management working
- [ ] Configuration classes implemented and tested

### ✅ Context7 MCP Integration
- [ ] Context7 client implemented
- [ ] Documentation retrieval working
- [ ] Topic extraction from subtasks functional

### ✅ Blender Integration
- [ ] Blender executor implemented
- [ ] Code wrapping and execution working  
- [ ] Screenshot capture functional
- [ ] Basic error handling in place

### ✅ Development Environment
- [ ] Logging configured with structured output
- [ ] Basic tests written and passing
- [ ] Code formatting and linting setup
- [ ] Git configuration and workflows

### ✅ Validation Tests
- [ ] Configuration loading tests pass
- [ ] Context7 integration tests pass
- [ ] Blender executor tests pass
- [ ] All dependencies install successfully

---

## Next Steps (Phase 2)

Once Phase 1 is complete, Phase 2 will focus on:

1. **Agent Implementation**: Build the five core agents (Planner, Retrieval, Coding, Critic, Verification)
2. **LangGraph Integration**: Set up the workflow orchestration
3. **State Management**: Implement the workflow state system
4. **Agent Communication**: Establish inter-agent communication protocols

This foundation provides the infrastructure needed to build the multi-agent system in subsequent phases.