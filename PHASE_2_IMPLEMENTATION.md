# Phase 2 Implementation: Multi-Agent System Development

## Overview

Phase 2 focuses on implementing the core multi-agent system for LL3M, building upon the solid foundation established in Phase 1. This phase implements the 5 specialized agents and their orchestration through LangGraph workflows.

## Objectives

- Implement all 5 specialized agents (Planner, Retrieval, Coding, Critic, Verification)
- Create LangGraph workflow orchestration
- Establish agent communication protocols
- Implement the three-phase generation pipeline
- Add comprehensive testing for agent interactions
- Create evaluation metrics for agent performance

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow Engine                    │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Initial Creation                                     │
│  ┌─────────────┐ → ┌─────────────┐ → ┌─────────────┐ → Exec   │
│  │   Planner   │   │  Retrieval  │   │   Coding    │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                 │
│  Phase 2: Automatic Refinement                                 │
│  ┌─────────────┐ → ┌─────────────┐ ← ┌─────────────┐           │
│  │   Critic    │   │Verification │   │   Coding    │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                 │
│  Phase 3: User-Guided Refinement                               │
│  User Feedback → Agent Pipeline → Iterative Improvement        │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Task 1: Agent Interface Standardization
**Duration**: 2 days
**Priority**: Critical

#### 1.1 Enhanced Agent Protocol
```python
from typing import Protocol, runtime_checkable, Any, Dict
from abc import abstractmethod
import structlog

@runtime_checkable
class LL3MAgent(Protocol):
    """Enhanced protocol for all LL3M agents with logging and metrics."""
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return agent type identifier."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable agent name."""
        ...
    
    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""
        ...
    
    @abstractmethod
    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input state before processing."""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Return agent performance metrics."""
        ...
```

#### 1.2 Enhanced Base Agent
```python
import asyncio
import time
from typing import Dict, Any, Optional
from abc import abstractmethod

import structlog
from openai import AsyncOpenAI

from ..utils.types import AgentResponse, AgentType, WorkflowState
from ..utils.config import settings

logger = structlog.get_logger(__name__)

class EnhancedBaseAgent:
    """Enhanced base class with OpenAI integration, metrics, and error handling."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced base agent."""
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.max_retries = config.get("max_retries", 3)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0,
        }
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def make_openai_request(
        self,
        messages: list[Dict[str, str]],
        **kwargs
    ) -> str:
        """Make OpenAI API request with retry logic and error handling."""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                
                # Update metrics
                execution_time = time.time() - start_time
                self._update_metrics(execution_time, response.usage.total_tokens)
                
                self.logger.info(
                    "OpenAI request successful",
                    attempt=attempt + 1,
                    execution_time=execution_time,
                    tokens_used=response.usage.total_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                self.logger.warning(
                    "OpenAI request failed",
                    attempt=attempt + 1,
                    error=str(e),
                    max_retries=self.max_retries
                )
                
                if attempt == self.max_retries - 1:
                    self.metrics["failed_requests"] += 1
                    raise
                    
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    def _update_metrics(self, execution_time: float, tokens_used: int) -> None:
        """Update agent performance metrics."""
        self.metrics["successful_requests"] += 1
        self.metrics["total_tokens_used"] += tokens_used
        
        # Update rolling average response time
        total_successful = self.metrics["successful_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_successful - 1) + execution_time) / total_successful
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current agent metrics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0.0
            ),
            "agent_type": self.agent_type.value,
            "model": self.model
        }
    
    async def validate_input(self, state: WorkflowState) -> bool:
        """Default input validation - can be overridden by subclasses."""
        return state.prompt is not None and len(state.prompt.strip()) > 0
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return agent type identifier."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable agent name."""
        pass
    
    @abstractmethod
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Process workflow state and return response."""
        pass
```

#### Success Criteria
- ✅ Enhanced agent protocol with validation and metrics
- ✅ Base agent class with OpenAI integration
- ✅ Retry logic and error handling
- ✅ Comprehensive metrics collection
- ✅ Structured logging throughout

### Task 2: Planner Agent Implementation
**Duration**: 3 days
**Priority**: Critical

#### 2.1 Task Decomposition System
```python
from typing import List, Dict, Any
import json
from dataclasses import dataclass

from ..utils.types import SubTask, TaskType, AgentResponse, AgentType, WorkflowState
from .base import EnhancedBaseAgent

@dataclass
class TaskDecompositionPrompt:
    """Template for task decomposition prompts."""
    
    SYSTEM_PROMPT = """You are a 3D modeling task planner specializing in Blender workflows.
    
    Your role is to analyze natural language prompts and decompose them into structured subtasks
    for 3D asset creation. You understand geometry, materials, lighting, scene setup, and animation.
    
    For each subtask, determine:
    1. Task type (geometry, material, lighting, scene_setup, animation)
    2. Priority (1-5, where 1 is highest priority)
    3. Dependencies (which tasks must complete first)
    4. Specific parameters needed for execution
    
    Return tasks in optimal execution order with clear, actionable descriptions."""
    
    USER_TEMPLATE = """Analyze this prompt and create a detailed task breakdown:
    
    Prompt: "{prompt}"
    
    Return your response as a JSON array of tasks with this structure:
    {{
      "tasks": [
        {{
          "id": "task-1",
          "type": "geometry|material|lighting|scene_setup|animation",
          "description": "Clear, specific description of what to create",
          "priority": 1-5,
          "dependencies": ["task-id-1", "task-id-2"],
          "parameters": {{"key": "value", ...}}
        }}
      ],
      "reasoning": "Brief explanation of the task breakdown approach"
    }}
    
    Guidelines:
    - Create 1-8 tasks depending on complexity
    - Be specific about shapes, colors, materials, positions
    - Consider proper task ordering and dependencies
    - Include realistic parameters for Blender operations
    """

class PlannerAgent(EnhancedBaseAgent):
    """Decomposes natural language prompts into structured 3D modeling subtasks."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize planner agent."""
        super().__init__(config)
        self.task_decomposer = TaskDecompositionPrompt()
    
    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.PLANNER
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return "Task Planner"
    
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Decompose prompt into structured subtasks."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate input
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message="Invalid input: prompt is required",
                    execution_time=0.0
                )
            
            self.logger.info("Starting task decomposition", prompt=state.prompt[:100])
            
            # Create decomposition prompt
            messages = [
                {"role": "system", "content": self.task_decomposer.SYSTEM_PROMPT},
                {"role": "user", "content": self.task_decomposer.USER_TEMPLATE.format(
                    prompt=state.prompt
                )}
            ]
            
            # Get LLM response
            response_text = await self.make_openai_request(messages)
            
            # Parse JSON response
            try:
                response_data = json.loads(response_text)
                tasks_data = response_data.get("tasks", [])
                reasoning = response_data.get("reasoning", "")
            except json.JSONDecodeError as e:
                self.logger.error("Failed to parse JSON response", error=str(e))
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message=f"Failed to parse response: {str(e)}",
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # Convert to SubTask objects
            subtasks = []
            for i, task_data in enumerate(tasks_data):
                try:
                    subtask = SubTask(
                        id=task_data.get("id", f"task-{i+1}"),
                        type=TaskType(task_data["type"]),
                        description=task_data["description"],
                        priority=task_data.get("priority", 1),
                        dependencies=task_data.get("dependencies", []),
                        parameters=task_data.get("parameters", {})
                    )
                    subtasks.append(subtask)
                except (KeyError, ValueError) as e:
                    self.logger.warning("Invalid task data", task_data=task_data, error=str(e))
                    continue
            
            if not subtasks:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data=[],
                    message="No valid subtasks generated",
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # Sort by priority and dependencies
            ordered_subtasks = self._order_tasks_by_dependencies(subtasks)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.info(
                "Task decomposition completed",
                num_tasks=len(ordered_subtasks),
                execution_time=execution_time
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=ordered_subtasks,
                message=f"Generated {len(ordered_subtasks)} subtasks: {reasoning}",
                execution_time=execution_time,
                metadata={"reasoning": reasoning, "original_prompt": state.prompt}
            )
            
        except Exception as e:
            self.logger.error("Task decomposition failed", error=str(e))
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data=[],
                message=f"Decomposition failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    def _order_tasks_by_dependencies(self, tasks: List[SubTask]) -> List[SubTask]:
        """Order tasks respecting dependencies and priorities."""
        # Implementation of topological sort with priority consideration
        ordered = []
        remaining = {task.id: task for task in tasks}
        
        while remaining:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = []
            for task in remaining.values():
                if all(dep_id not in remaining for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or missing dependency - break it
                ready_tasks = [next(iter(remaining.values()))]
                self.logger.warning("Breaking circular or missing dependencies")
            
            # Sort ready tasks by priority (lower number = higher priority)
            ready_tasks.sort(key=lambda t: t.priority)
            
            # Add highest priority task
            next_task = ready_tasks[0]
            ordered.append(next_task)
            del remaining[next_task.id]
        
        return ordered
    
    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for planner agent."""
        if not state.prompt:
            return False
        
        # Check prompt length (minimum and maximum)
        prompt_length = len(state.prompt.strip())
        if prompt_length < 5 or prompt_length > 2000:
            return False
        
        return True
```

#### 2.2 Task Templates and Examples
```python
class TaskTemplates:
    """Predefined templates for common task patterns."""
    
    SIMPLE_OBJECT = {
        "pattern": r"\b(create|make|add)\s+(a|an)?\s*(\w+)\b",
        "template": [
            {
                "type": "geometry",
                "priority": 1,
                "description": "Create {object_type} geometry",
                "parameters": {"shape": "{object_type}"}
            }
        ]
    }
    
    OBJECT_WITH_MATERIAL = {
        "pattern": r"\b(\w+)\s+(\w+)\s+(cube|sphere|cylinder|plane)\b",
        "template": [
            {
                "type": "geometry",
                "priority": 1,
                "description": "Create {shape} geometry"
            },
            {
                "type": "material",
                "priority": 2,
                "dependencies": ["geometry"],
                "description": "Apply {color} {material_type} material"
            }
        ]
    }
```

#### Success Criteria
- ✅ Natural language → structured subtasks conversion
- ✅ Task dependency analysis and ordering
- ✅ Parameter extraction for Blender operations
- ✅ JSON response parsing with error handling
- ✅ Comprehensive test coverage (>90%)

### Task 3: Retrieval Agent Implementation
**Duration**: 3 days
**Priority**: Critical

#### 3.1 Context7 Documentation Retrieval
```python
import asyncio
from typing import List, Dict, Any, Optional
import re

from ..knowledge.context7_client import Context7RetrievalService
from ..utils.types import SubTask, AgentResponse, AgentType, WorkflowState
from .base import EnhancedBaseAgent

class RetrievalAgent(EnhancedBaseAgent):
    """Retrieves relevant Blender documentation using Context7 MCP."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize retrieval agent."""
        super().__init__(config)
        self.context7_service = Context7RetrievalService()
        self.documentation_cache: Dict[str, str] = {}
        self.topic_extraction_patterns = {
            TaskType.GEOMETRY: [
                r"mesh", r"primitive", r"vertex", r"face", r"edge",
                r"cube", r"sphere", r"cylinder", r"plane", r"torus"
            ],
            TaskType.MATERIAL: [
                r"material", r"shader", r"node", r"bsdf", r"texture",
                r"color", r"metallic", r"roughness", r"emission"
            ],
            TaskType.LIGHTING: [
                r"light", r"lamp", r"sun", r"area", r"point",
                r"energy", r"color", r"shadow", r"hdri"
            ],
            TaskType.SCENE_SETUP: [
                r"camera", r"render", r"scene", r"world", r"background"
            ],
            TaskType.ANIMATION: [
                r"keyframe", r"animation", r"timeline", r"frame"
            ]
        }
    
    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.RETRIEVAL
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return "Documentation Retrieval"
    
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Retrieve relevant documentation for subtasks."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data="",
                    message="Invalid input: subtasks are required",
                    execution_time=0.0
                )
            
            self.logger.info("Starting documentation retrieval", num_subtasks=len(state.subtasks))
            
            # Extract topics from subtasks
            topics = self._extract_topics_from_subtasks(state.subtasks)
            
            # Build search queries
            search_queries = self._build_search_queries(state.subtasks, topics)
            
            # Retrieve documentation concurrently
            documentation_parts = await self._retrieve_documentation_parallel(search_queries)
            
            # Combine and filter documentation
            combined_docs = self._combine_documentation(documentation_parts, state.subtasks)
            
            # Enhance with context-specific examples
            enhanced_docs = await self._enhance_with_examples(combined_docs, state.subtasks)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.info(
                "Documentation retrieval completed",
                doc_length=len(enhanced_docs),
                topics=topics,
                execution_time=execution_time
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=enhanced_docs,
                message=f"Retrieved documentation for {len(topics)} topics",
                execution_time=execution_time,
                metadata={
                    "topics": topics,
                    "search_queries": search_queries,
                    "doc_sections": len(documentation_parts)
                }
            )
            
        except Exception as e:
            self.logger.error("Documentation retrieval failed", error=str(e))
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data="",
                message=f"Retrieval failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    def _extract_topics_from_subtasks(self, subtasks: List[SubTask]) -> List[str]:
        """Extract documentation topics from subtasks."""
        topics = set()
        
        for subtask in subtasks:
            # Add task type as primary topic
            topics.add(subtask.type.value)
            
            # Extract specific keywords based on task type and description
            description_lower = subtask.description.lower()
            
            if subtask.type in self.topic_extraction_patterns:
                patterns = self.topic_extraction_patterns[subtask.type]
                for pattern in patterns:
                    if re.search(pattern, description_lower):
                        topics.add(pattern)
            
            # Extract from parameters
            for key, value in subtask.parameters.items():
                if isinstance(value, str):
                    topics.add(value.lower())
        
        return list(topics)
    
    def _build_search_queries(self, subtasks: List[SubTask], topics: List[str]) -> List[str]:
        """Build specific search queries for Context7."""
        queries = []
        
        # Topic-based queries
        for topic in topics:
            queries.append(f"blender python {topic} api")
        
        # Task-specific queries
        for subtask in subtasks:
            if subtask.type == TaskType.GEOMETRY:
                queries.append(f"bpy.ops.mesh.primitive {subtask.description}")
            elif subtask.type == TaskType.MATERIAL:
                queries.append(f"blender material nodes bsdf {subtask.description}")
            elif subtask.type == TaskType.LIGHTING:
                queries.append(f"bpy.ops.object.light_add {subtask.description}")
        
        return list(set(queries))  # Remove duplicates
    
    async def _retrieve_documentation_parallel(self, queries: List[str]) -> List[str]:
        """Retrieve documentation for multiple queries in parallel."""
        tasks = []
        
        for query in queries:
            # Check cache first
            if query in self.documentation_cache:
                tasks.append(asyncio.create_task(self._get_cached_docs(query)))
            else:
                tasks.append(asyncio.create_task(self._fetch_and_cache_docs(query)))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _get_cached_docs(self, query: str) -> str:
        """Get documentation from cache."""
        return self.documentation_cache[query]
    
    async def _fetch_and_cache_docs(self, query: str) -> str:
        """Fetch documentation and cache it."""
        try:
            response = await self.context7_service.retrieve_documentation([query])
            
            if response.success and response.data:
                self.documentation_cache[query] = response.data
                return response.data
            else:
                self.logger.warning("Failed to retrieve docs", query=query)
                return ""
        except Exception as e:
            self.logger.error("Documentation fetch error", query=query, error=str(e))
            return ""
    
    def _combine_documentation(self, doc_parts: List[str], subtasks: List[SubTask]) -> str:
        """Combine retrieved documentation into coherent guide."""
        # Filter out empty or error responses
        valid_docs = [doc for doc in doc_parts if isinstance(doc, str) and doc.strip()]
        
        if not valid_docs:
            return self._get_fallback_documentation(subtasks)
        
        # Structure documentation by sections
        sections = {
            "overview": "# Blender Python API Documentation\n\n",
            "basic_operations": "",
            "geometry": "",
            "materials": "",
            "lighting": "",
            "examples": ""
        }
        
        # Categorize and organize documentation
        for doc in valid_docs:
            if "mesh" in doc.lower() or "primitive" in doc.lower():
                sections["geometry"] += doc + "\n\n"
            elif "material" in doc.lower() or "bsdf" in doc.lower():
                sections["materials"] += doc + "\n\n"
            elif "light" in doc.lower() or "lamp" in doc.lower():
                sections["lighting"] += doc + "\n\n"
            else:
                sections["basic_operations"] += doc + "\n\n"
        
        # Combine sections
        combined = sections["overview"]
        
        for section_name, content in sections.items():
            if section_name != "overview" and content.strip():
                combined += f"## {section_name.replace('_', ' ').title()}\n\n{content}"
        
        return combined
    
    async def _enhance_with_examples(self, documentation: str, subtasks: List[SubTask]) -> str:
        """Enhance documentation with context-specific examples."""
        # Generate examples based on subtasks
        examples_section = "\n\n## Context-Specific Examples\n\n"
        
        for subtask in subtasks:
            if subtask.type == TaskType.GEOMETRY:
                examples_section += f"### {subtask.description}\n"
                examples_section += self._generate_geometry_example(subtask)
            elif subtask.type == TaskType.MATERIAL:
                examples_section += f"### {subtask.description}\n"
                examples_section += self._generate_material_example(subtask)
        
        return documentation + examples_section
    
    def _generate_geometry_example(self, subtask: SubTask) -> str:
        """Generate geometry creation example."""
        shape = subtask.parameters.get("shape", "cube").lower()
        location = subtask.parameters.get("location", [0, 0, 0])
        
        return f"""```python
# Create {shape}
bpy.ops.mesh.primitive_{shape}_add(location={location})
obj = bpy.context.active_object
obj.name = "{subtask.id}"
```

"""
    
    def _generate_material_example(self, subtask: SubTask) -> str:
        """Generate material creation example."""
        color = subtask.parameters.get("color", [0.8, 0.2, 0.2])
        
        return f"""```python
# Create material for {subtask.description}
material = bpy.data.materials.new(name="{subtask.id}_material")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = {color + [1.0]}

# Assign to active object
bpy.context.active_object.data.materials.append(material)
```

"""
    
    def _get_fallback_documentation(self, subtasks: List[SubTask]) -> str:
        """Provide fallback documentation when Context7 fails."""
        return """# Basic Blender Python API Reference

## Geometry Creation
```python
import bpy

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Add primitives
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
```

## Material Creation
```python
# Create material
material = bpy.data.materials.new(name="MyMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)
```
"""
    
    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for retrieval agent."""
        return state.subtasks is not None and len(state.subtasks) > 0
```

#### Success Criteria
- ✅ Context7 MCP integration working
- ✅ Parallel documentation retrieval
- ✅ Documentation caching system
- ✅ Topic extraction and query building
- ✅ Fallback documentation for failures

### Task 4: Coding Agent Implementation
**Duration**: 4 days  
**Priority**: Critical

#### 4.1 Code Generation System
```python
import asyncio
import json
from typing import List, Dict, Any, Optional
import re

from ..blender.templates import (
    GEOMETRY_TEMPLATES, MATERIAL_TEMPLATES, 
    LIGHTING_TEMPLATES, SCENE_TEMPLATES, MODIFIER_TEMPLATES
)
from ..utils.types import SubTask, TaskType, AgentResponse, AgentType, WorkflowState
from .base import EnhancedBaseAgent

class CodeGenerationPrompt:
    """Templates for code generation prompts."""
    
    SYSTEM_PROMPT = """You are an expert Blender Python programmer specializing in procedural 3D asset creation.

Your role is to generate clean, efficient, and executable Blender Python code based on:
1. Structured subtasks with specific requirements
2. Relevant Blender API documentation
3. Code templates and best practices

Code Requirements:
- Use only the Blender Python API (bpy module)
- Generate modular, readable code with proper error handling
- Include comments explaining key operations
- Follow Blender best practices for object creation and manipulation
- Ensure objects are properly named and organized
- Handle edge cases and provide fallbacks

Code Structure:
- Import statements at the top
- Scene setup and cleanup
- Object creation and modification
- Material and lighting setup
- Final scene organization
"""
    
    USER_TEMPLATE = """Generate Blender Python code for these subtasks:

Subtasks:
{subtasks_json}

Available Documentation:
{documentation}

Requirements:
- Generate complete, executable Python code
- Follow the subtask order and dependencies
- Use proper Blender API calls
- Include error handling and logging
- Make code modular and well-commented
- Ensure reproducible results

Return only the Python code, no explanations."""

class CodingAgent(EnhancedBaseAgent):
    """Generates executable Blender Python code from subtasks and documentation."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize coding agent."""
        super().__init__(config)
        self.code_generator = CodeGenerationPrompt()
        self.templates = {
            TaskType.GEOMETRY: GEOMETRY_TEMPLATES,
            TaskType.MATERIAL: MATERIAL_TEMPLATES,
            TaskType.LIGHTING: LIGHTING_TEMPLATES,
            TaskType.SCENE_SETUP: SCENE_TEMPLATES,
        }
    
    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.CODING
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return "Code Generator"
    
    async def process(self, state: WorkflowState) -> AgentResponse:
        """Generate Blender Python code from subtasks and documentation."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data="",
                    message="Invalid input: subtasks and documentation required",
                    execution_time=0.0
                )
            
            self.logger.info("Starting code generation", num_subtasks=len(state.subtasks))
            
            # Prepare subtasks data for LLM
            subtasks_json = self._prepare_subtasks_for_llm(state.subtasks)
            
            # Create generation prompt
            messages = [
                {"role": "system", "content": self.code_generator.SYSTEM_PROMPT},
                {"role": "user", "content": self.code_generator.USER_TEMPLATE.format(
                    subtasks_json=subtasks_json,
                    documentation=state.documentation
                )}
            ]
            
            # Generate code with LLM
            raw_code = await self.make_openai_request(messages)
            
            # Clean and validate generated code
            clean_code = self._clean_generated_code(raw_code)
            
            # Enhance code with templates and best practices
            enhanced_code = self._enhance_with_templates(clean_code, state.subtasks)
            
            # Add standard imports and setup
            final_code = self._add_standard_setup(enhanced_code)
            
            # Validate code structure
            validation_result = self._validate_code_structure(final_code)
            
            if not validation_result["valid"]:
                self.logger.warning("Generated code failed validation", 
                                  issues=validation_result["issues"])
                # Try fallback template-based generation
                final_code = self._generate_fallback_code(state.subtasks)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.info(
                "Code generation completed",
                code_length=len(final_code),
                execution_time=execution_time
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=final_code,
                message=f"Generated {len(final_code)} characters of Blender Python code",
                execution_time=execution_time,
                metadata={
                    "code_lines": len(final_code.split('\n')),
                    "validation": validation_result
                }
            )
            
        except Exception as e:
            self.logger.error("Code generation failed", error=str(e))
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data="",
                message=f"Code generation failed: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    def _prepare_subtasks_for_llm(self, subtasks: List[SubTask]) -> str:
        """Prepare subtasks data in JSON format for LLM consumption."""
        subtasks_data = []
        
        for subtask in subtasks:
            subtasks_data.append({
                "id": subtask.id,
                "type": subtask.type.value,
                "description": subtask.description,
                "priority": subtask.priority,
                "dependencies": subtask.dependencies,
                "parameters": subtask.parameters
            })
        
        return json.dumps(subtasks_data, indent=2)
    
    def _clean_generated_code(self, raw_code: str) -> str:
        """Clean and format generated code."""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*\n?', '', raw_code)
        code = re.sub(r'```\s*$', '', code)
        
        # Remove explanatory text before/after code
        lines = code.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find first import or bpy statement
        for i, line in enumerate(lines):
            if ('import ' in line or 'bpy.' in line) and not line.strip().startswith('#'):
                start_idx = i
                break
        
        # Take code from first import to end
        cleaned_lines = lines[start_idx:end_idx]
        
        # Remove trailing explanatory text
        while cleaned_lines and not any(
            keyword in cleaned_lines[-1] for keyword in ['bpy.', 'import', 'def ', 'class ', '=', 'if ', 'for ']
        ) and not cleaned_lines[-1].strip().startswith('#'):
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def _enhance_with_templates(self, code: str, subtasks: List[SubTask]) -> str:
        """Enhance generated code with template-based improvements."""
        enhanced_code = code
        
        # Add template-based enhancements for missing functionality
        for subtask in subtasks:
            if subtask.type in self.templates:
                template_additions = self._get_template_enhancements(subtask)
                if template_additions:
                    enhanced_code += f"\n\n# Template enhancement for {subtask.id}\n"
                    enhanced_code += template_additions
        
        return enhanced_code
    
    def _get_template_enhancements(self, subtask: SubTask) -> str:
        """Get template-based code enhancements for a subtask."""
        templates = self.templates.get(subtask.type, {})
        
        # Extract relevant templates based on subtask description
        relevant_templates = []
        description_lower = subtask.description.lower()
        
        for template_name, template_code in templates.items():
            if template_name.lower() in description_lower:
                relevant_templates.append(template_code)
        
        if relevant_templates:
            # Format templates with subtask parameters
            formatted_templates = []
            for template in relevant_templates:
                try:
                    formatted = template.format(**subtask.parameters)
                    formatted_templates.append(formatted)
                except KeyError:
                    # Template requires parameters not available
                    formatted_templates.append(template)
            
            return '\n'.join(formatted_templates)
        
        return ""
    
    def _add_standard_setup(self, code: str) -> str:
        """Add standard imports and setup code."""
        setup_code = """import bpy
import bmesh
import mathutils
from mathutils import Vector

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Ensure we're in object mode
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

"""
        
        # Check if code already has imports/setup
        if 'import bpy' not in code:
            return setup_code + code
        else:
            return code
    
    def _validate_code_structure(self, code: str) -> Dict[str, Any]:
        """Validate the structure and safety of generated code."""
        issues = []
        
        # Check for required imports
        if 'import bpy' not in code:
            issues.append("Missing bpy import")
        
        # Check for unsafe operations
        unsafe_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'import\s+os',
            r'import\s+subprocess',
            r'__import__',
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Unsafe operation detected: {pattern}")
        
        # Check for basic Blender operations
        if 'bpy.ops.' not in code and 'bpy.data.' not in code:
            issues.append("No Blender operations found")
        
        # Validate Python syntax
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _generate_fallback_code(self, subtasks: List[SubTask]) -> str:
        """Generate fallback code using templates when LLM generation fails."""
        fallback_code = """import bpy
import bmesh

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

"""
        
        for subtask in subtasks:
            if subtask.type == TaskType.GEOMETRY:
                fallback_code += self._generate_geometry_fallback(subtask)
            elif subtask.type == TaskType.MATERIAL:
                fallback_code += self._generate_material_fallback(subtask)
            elif subtask.type == TaskType.LIGHTING:
                fallback_code += self._generate_lighting_fallback(subtask)
            
            fallback_code += "\n"
        
        return fallback_code
    
    def _generate_geometry_fallback(self, subtask: SubTask) -> str:
        """Generate fallback geometry code."""
        shape = subtask.parameters.get("shape", "cube")
        location = subtask.parameters.get("location", [0, 0, 0])
        
        return f"""
# Create {subtask.description}
bpy.ops.mesh.primitive_{shape}_add(location={location})
obj = bpy.context.active_object
obj.name = "{subtask.id}"
"""
    
    def _generate_material_fallback(self, subtask: SubTask) -> str:
        """Generate fallback material code."""
        color = subtask.parameters.get("color", [0.8, 0.2, 0.2, 1.0])
        
        return f"""
# Create material for {subtask.description}
material = bpy.data.materials.new(name="{subtask.id}_material")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = {color}

# Assign to active object
if bpy.context.active_object:
    bpy.context.active_object.data.materials.append(material)
"""
    
    def _generate_lighting_fallback(self, subtask: SubTask) -> str:
        """Generate fallback lighting code."""
        light_type = subtask.parameters.get("type", "SUN")
        location = subtask.parameters.get("location", [5, 5, 10])
        energy = subtask.parameters.get("energy", 3.0)
        
        return f"""
# Create light for {subtask.description}
bpy.ops.object.light_add(type='{light_type}', location={location})
light = bpy.context.active_object
light.data.energy = {energy}
light.name = "{subtask.id}_light"
"""
    
    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for coding agent."""
        if not state.subtasks or len(state.subtasks) == 0:
            return False
        
        if not state.documentation or len(state.documentation.strip()) == 0:
            self.logger.warning("No documentation provided, will use templates")
            # Still valid, will use fallback templates
        
        return True
```

#### Success Criteria
- ✅ Natural language → executable Blender Python code
- ✅ Template integration and enhancement
- ✅ Code validation and safety checking
- ✅ Fallback generation for LLM failures
- ✅ Proper error handling and logging

## Workflow Integration

### Task 5: LangGraph Workflow Implementation
**Duration**: 3 days
**Priority**: Critical

#### 5.1 Workflow State Management
```python
from langgraph import StateGraph, END
from typing import Literal

from ..utils.types import WorkflowState, AgentType
from .planner import PlannerAgent
from .retrieval import RetrievalAgent
from .coding import CodingAgent
from ..blender.executor import BlenderExecutor

async def planner_node(state: WorkflowState) -> WorkflowState:
    """Execute planner agent."""
    planner = PlannerAgent(settings.get_agent_config("planner"))
    response = await planner.process(state)
    
    if response.success:
        state.subtasks = response.data
    else:
        state.error_message = response.message
        state.should_continue = False
    
    return state

async def retrieval_node(state: WorkflowState) -> WorkflowState:
    """Execute retrieval agent."""
    retrieval = RetrievalAgent(settings.get_agent_config("retrieval"))
    response = await retrieval.process(state)
    
    if response.success:
        state.documentation = response.data
    else:
        state.error_message = response.message
        state.should_continue = False
    
    return state

async def coding_node(state: WorkflowState) -> WorkflowState:
    """Execute coding agent."""
    coding = CodingAgent(settings.get_agent_config("coding"))
    response = await coding.process(state)
    
    if response.success:
        state.generated_code = response.data
    else:
        state.error_message = response.message
        state.should_continue = False
    
    return state

async def execution_node(state: WorkflowState) -> WorkflowState:
    """Execute generated code in Blender."""
    executor = BlenderExecutor()
    
    try:
        result = await executor.execute_code(
            state.generated_code,
            asset_name=f"asset_{int(asyncio.get_event_loop().time())}"
        )
        
        state.execution_result = result
        
        if result.success:
            state.asset_metadata = AssetMetadata(
                id=f"asset_{uuid.uuid4()}",
                prompt=state.prompt,
                file_path=result.asset_path,
                screenshot_path=result.screenshot_path,
                subtasks=state.subtasks
            )
        else:
            state.error_message = f"Execution failed: {'; '.join(result.errors)}"
            
    except Exception as e:
        state.error_message = f"Execution error: {str(e)}"
        state.should_continue = False
    
    return state

def should_continue(state: WorkflowState) -> Literal["end", "continue"]:
    """Determine if workflow should continue."""
    if not state.should_continue or state.error_message:
        return "end"
    return "continue"

def create_initial_workflow() -> StateGraph:
    """Create the initial creation workflow."""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node) 
    workflow.add_node("coding", coding_node)
    workflow.add_node("execution", execution_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {"continue": "retrieval", "end": END}
    )
    
    workflow.add_conditional_edges(
        "retrieval", 
        should_continue,
        {"continue": "coding", "end": END}
    )
    
    workflow.add_conditional_edges(
        "coding",
        should_continue, 
        {"continue": "execution", "end": END}
    )
    
    workflow.add_edge("execution", END)
    
    return workflow.compile()
```

### Task 6: Testing and Validation
**Duration**: 3 days
**Priority**: High

#### 6.1 Agent Unit Tests
```python
# tests/unit/test_planner_agent.py
import pytest
from unittest.mock import AsyncMock, patch

from src.agents.planner import PlannerAgent
from src.utils.types import WorkflowState, TaskType

class TestPlannerAgent:
    @pytest.fixture
    def planner(self):
        config = {"model": "gpt-4", "temperature": 0.7}
        return PlannerAgent(config)
    
    @pytest.mark.asyncio
    async def test_simple_object_decomposition(self, planner):
        """Test decomposition of simple object prompt."""
        state = WorkflowState(prompt="Create a red cube")
        
        with patch.object(planner, 'make_openai_request') as mock_request:
            mock_request.return_value = '''
            {
                "tasks": [
                    {
                        "id": "geometry-1",
                        "type": "geometry", 
                        "description": "Create cube geometry",
                        "priority": 1,
                        "parameters": {"shape": "cube", "location": [0, 0, 0]}
                    },
                    {
                        "id": "material-1",
                        "type": "material",
                        "description": "Apply red material",
                        "priority": 2,
                        "dependencies": ["geometry-1"],
                        "parameters": {"color": [0.8, 0.2, 0.2]}
                    }
                ],
                "reasoning": "Simple object with material"
            }
            '''
            
            response = await planner.process(state)
            
            assert response.success
            assert len(response.data) == 2
            assert response.data[0].type == TaskType.GEOMETRY
            assert response.data[1].type == TaskType.MATERIAL
            assert response.data[1].dependencies == ["geometry-1"]
```

#### 6.2 Integration Tests
```python
# tests/integration/test_agent_workflow.py
import pytest
from src.workflow.graph import create_initial_workflow
from src.utils.types import WorkflowState

@pytest.mark.asyncio
@pytest.mark.integration
async def test_complete_workflow_execution():
    """Test complete workflow from prompt to asset."""
    workflow = create_initial_workflow()
    
    initial_state = WorkflowState(
        prompt="Create a blue metallic sphere"
    )
    
    final_state = await workflow.ainvoke(initial_state)
    
    assert final_state.subtasks is not None
    assert len(final_state.subtasks) > 0
    assert final_state.documentation != ""
    assert final_state.generated_code != ""
    assert final_state.execution_result is not None
    
    if final_state.execution_result.success:
        assert final_state.asset_metadata is not None
        assert final_state.asset_metadata.file_path is not None
```

## Success Criteria

### Phase 2 Completion Requirements

#### Agent Implementation (70%)
- ✅ All 5 agents implemented with proper interfaces
- ✅ OpenAI integration with retry logic and error handling
- ✅ Comprehensive metrics and logging
- ✅ Input validation and safety checks
- ✅ Template-based fallbacks for failures

#### Workflow Orchestration (20%)
- ✅ LangGraph workflow implementation
- ✅ State management between agents
- ✅ Error handling and graceful failures
- ✅ Conditional flow control

#### Testing & Quality (10%)
- ✅ >85% test coverage for all agents
- ✅ Integration tests for workflows
- ✅ Performance benchmarks established
- ✅ Error condition testing

## Timeline

**Week 1 (Days 1-7)**:
- Task 1: Agent Interface Standardization (Days 1-2)
- Task 2: Planner Agent Implementation (Days 3-5)  
- Task 3: Retrieval Agent Implementation (Days 6-7)

**Week 2 (Days 8-14)**:
- Task 4: Coding Agent Implementation (Days 8-11)
- Task 5: LangGraph Workflow Implementation (Days 12-14)

**Week 3 (Days 15-21)**:
- Task 6: Testing and Validation (Days 15-17)
- Integration testing and bug fixes (Days 18-21)

## Risk Mitigation

### Technical Risks
- **OpenAI API Rate Limits**: Implement exponential backoff and request queuing
- **Context7 MCP Failures**: Fallback documentation and template-based generation
- **Code Generation Quality**: Multiple validation layers and template enhancement

### Quality Risks  
- **Agent Communication**: Comprehensive integration testing
- **State Management**: Detailed workflow testing with various scenarios
- **Error Handling**: Extensive error condition testing

## Next Steps After Phase 2

Phase 3 will focus on:
1. Critic Agent implementation (visual analysis)
2. Verification Agent implementation  
3. Automatic refinement workflow
4. User-guided refinement system
5. Performance optimization

This implementation plan ensures a robust, production-ready multi-agent system with proper error handling, comprehensive testing, and maintainable code structure.