# Phase 6: Testing & Validation Implementation Plan

## Overview
This document outlines the implementation plan for Phase 6 of LL3M: Testing & Validation. This phase focuses on comprehensive testing, quality assurance, evaluation metrics, and production readiness validation following the highest standards from GOOD_PRACTICES.md.

## Phase 6 Objectives
- **Comprehensive Test Coverage**: Achieve 90%+ test coverage across all components
- **Integration Testing**: End-to-end workflow validation and multi-agent coordination
- **Performance Benchmarking**: Asset quality metrics and system performance evaluation
- **Production Readiness**: Edge case handling, error recovery, and scalability validation
- **Quality Assurance**: Automated testing pipeline and continuous validation

## Architecture Overview

### Testing Strategy Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Unit Tests    │    │ Integration     │    │  End-to-End     │
│   (90%+ Cov)    │    │   Tests         │    │    Tests        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Performance   │
                    │   Benchmarks    │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Quality       │    │   Production    │    │   Evaluation    │
│   Validation    │    │   Testing       │    │   Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Implementation Tasks

### Task 1: Enhanced Unit Testing Framework
**Duration**: 2-3 days

#### 1.1 Test Infrastructure Enhancement
```python
# tests/framework/base.py
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional, List
import structlog

from src.utils.types import WorkflowState, AgentResponse, ExecutionResult
from src.utils.config import settings

class BaseTestCase:
    """Enhanced base test case with comprehensive utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path, caplog):
        """Set up isolated test environment."""
        self.tmp_path = tmp_path
        self.caplog = caplog
        
        # Create test directories
        self.test_assets_dir = tmp_path / "assets"
        self.test_output_dir = tmp_path / "outputs"
        self.test_cache_dir = tmp_path / "cache"
        
        for directory in [self.test_assets_dir, self.test_output_dir, self.test_cache_dir]:
            directory.mkdir(exist_ok=True)
        
        # Mock settings for testing
        self.original_settings = {}
        test_settings = {
            'app.output_directory': str(self.test_output_dir),
            'app.cache_directory': str(self.test_cache_dir),
            'blender.path': '/usr/bin/blender',
            'blender.timeout': 30,
            'openai.api_key': 'test_key'
        }
        
        for key, value in test_settings.items():
            self.original_settings[key] = getattr(settings, key, None)
            setattr(settings, key, value)
        
        yield
        
        # Cleanup
        for key, value in self.original_settings.items():
            setattr(settings, key, value)
    
    def create_mock_workflow_state(self, **kwargs) -> WorkflowState:
        """Create mock workflow state with sensible defaults."""
        defaults = {
            'prompt': 'Create a red cube',
            'original_prompt': 'Create a red cube',
            'subtasks': [],
            'documentation': '',
            'generated_code': '',
            'execution_result': None,
            'should_continue': True,
            'error_message': '',
            'refinement_iterations': 0
        }
        defaults.update(kwargs)
        return WorkflowState(**defaults)
    
    def create_mock_execution_result(self, success: bool = True, **kwargs) -> ExecutionResult:
        """Create mock execution result."""
        defaults = {
            'success': success,
            'asset_path': str(self.test_assets_dir / 'test_asset.blend') if success else None,
            'screenshot_path': str(self.test_assets_dir / 'screenshot.png') if success else None,
            'logs': ['Test execution log'],
            'errors': [] if success else ['Test error'],
            'execution_time': 2.5,
            'metadata': {}
        }
        defaults.update(kwargs)
        return ExecutionResult(**defaults)
    
    def create_test_asset_files(self) -> Dict[str, Path]:
        """Create test asset files."""
        files = {}
        
        # Create mock .blend file
        blend_file = self.test_assets_dir / "test_asset.blend"
        blend_file.write_bytes(b"BLENDER_TEST_DATA")
        files['blend'] = blend_file
        
        # Create mock screenshot
        screenshot_file = self.test_assets_dir / "screenshot.png"
        screenshot_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"MOCK_PNG_DATA" * 10)
        files['screenshot'] = screenshot_file
        
        # Create mock OBJ file
        obj_file = self.test_assets_dir / "test_asset.obj"
        obj_file.write_text("# Mock OBJ file\nv 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n")
        files['obj'] = obj_file
        
        return files

# tests/framework/mocks.py
class MockOpenAIClient:
    """Enhanced OpenAI client mock with realistic responses."""
    
    def __init__(self):
        self.call_history = []
        self.response_templates = {
            'planner': self._planner_response,
            'coding': self._coding_response,
            'critic': self._critic_response
        }
    
    async def create_completion(self, messages: List[Dict], **kwargs) -> str:
        """Mock completion creation with context-aware responses."""
        self.call_history.append({
            'messages': messages,
            'kwargs': kwargs,
            'timestamp': asyncio.get_event_loop().time()
        })
        
        # Determine response type based on message content
        content = str(messages).lower()
        if 'planner' in content or 'subtask' in content:
            return self.response_templates['planner'](messages)
        elif 'code' in content or 'blender' in content:
            return self.response_templates['coding'](messages)
        elif 'critic' in content or 'visual' in content:
            return self.response_templates['critic'](messages)
        
        return "Mock AI response"
    
    def _planner_response(self, messages: List[Dict]) -> str:
        """Generate realistic planner response."""
        return '''[
            {
                "id": "task_001",
                "type": "geometry",
                "description": "Create basic cube geometry",
                "parameters": {"size": 2.0, "location": [0, 0, 0]},
                "priority": 1
            },
            {
                "id": "task_002", 
                "type": "material",
                "description": "Apply red material",
                "parameters": {"color": [0.8, 0.2, 0.2], "roughness": 0.5},
                "priority": 2
            }
        ]'''
    
    def _coding_response(self, messages: List[Dict]) -> str:
        """Generate realistic coding response."""
        return '''
import bpy
import bmesh

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create cube geometry
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "RedCube"

# Create and apply material
material = bpy.data.materials.new(name="RedMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
bsdf.inputs[0].default_value = (0.8, 0.2, 0.2, 1.0)  # Base Color
bsdf.inputs[7].default_value = 0.5  # Roughness

cube.data.materials.append(material)
'''
    
    def _critic_response(self, messages: List[Dict]) -> str:
        """Generate realistic critic response."""
        return '''{
            "overall_score": 8.5,
            "visual_quality": {"score": 8.0, "notes": "Good color and shape"},
            "geometry": {"score": 9.0, "notes": "Clean topology"},
            "materials": {"score": 8.0, "notes": "Appropriate material settings"},
            "lighting": {"score": 7.0, "notes": "Could benefit from better lighting"},
            "composition": {"score": 8.5, "notes": "Well centered"},
            "requirements_match": {"score": 9.0, "notes": "Matches red cube requirement"},
            "needs_refinement": false,
            "critical_issues": [],
            "improvement_suggestions": ["Consider adding subtle lighting variations"],
            "refinement_priority": "low"
        }'''

class MockBlenderExecutor:
    """Enhanced Blender executor mock with realistic behavior."""
    
    def __init__(self):
        self.execution_history = []
        self.should_succeed = True
        self.execution_time = 2.5
    
    async def execute_code(self, code: str, **kwargs) -> ExecutionResult:
        """Mock code execution with configurable outcomes."""
        execution_start = asyncio.get_event_loop().time()
        
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        execution_record = {
            'code': code,
            'kwargs': kwargs,
            'timestamp': execution_start,
            'success': self.should_succeed
        }
        self.execution_history.append(execution_record)
        
        if self.should_succeed:
            return ExecutionResult(
                success=True,
                asset_path="/tmp/test_asset.blend",
                screenshot_path="/tmp/screenshot.png",
                logs=[f"Executed {len(code)} characters of code"],
                errors=[],
                execution_time=self.execution_time,
                metadata={'mock_execution': True}
            )
        else:
            return ExecutionResult(
                success=False,
                asset_path=None,
                screenshot_path=None,
                logs=["Execution started"],
                errors=["Mock execution error"],
                execution_time=self.execution_time,
                metadata={'mock_execution': True}
            )
    
    def set_failure_mode(self, should_fail: bool = True):
        """Configure executor to fail or succeed."""
        self.should_succeed = not should_fail
    
    def set_execution_time(self, time_seconds: float):
        """Configure execution time."""
        self.execution_time = time_seconds

# tests/framework/fixtures.py
@pytest.fixture
def mock_openai_client():
    """Provide mock OpenAI client."""
    return MockOpenAIClient()

@pytest.fixture
def mock_blender_executor():
    """Provide mock Blender executor."""
    return MockBlenderExecutor()

@pytest.fixture
def context7_mock_responses():
    """Mock responses for Context7 MCP queries."""
    return {
        'bpy.ops.mesh': '''
        bpy.ops.mesh module provides mesh operations:
        
        primitive_cube_add(size=2.0, location=(0,0,0))
        - Creates a cube primitive
        - size: Scale factor
        - location: 3D coordinates
        
        primitive_uv_sphere_add(radius=1.0, location=(0,0,0))
        - Creates a UV sphere
        - radius: Sphere radius
        - location: 3D coordinates
        ''',
        
        'bpy.data.materials': '''
        bpy.data.materials provides material management:
        
        new(name="MaterialName")
        - Creates new material
        - Returns Material object
        
        Material properties:
        - use_nodes: Enable node-based materials
        - node_tree: Access to material node tree
        ''',
        
        'bmesh': '''
        bmesh module for mesh editing:
        
        new() - Create new bmesh instance
        from_mesh(mesh) - Create from existing mesh
        to_mesh(mesh) - Apply changes to mesh
        '''
    }

@pytest.fixture
async def mock_context7_client(context7_mock_responses):
    """Mock Context7 MCP client."""
    class MockContext7Client:
        def __init__(self):
            self.responses = context7_mock_responses
            self.query_history = []
        
        async def query_documentation(self, query: str) -> str:
            self.query_history.append(query)
            
            # Return relevant documentation based on query
            for key, response in self.responses.items():
                if key.lower() in query.lower():
                    return response
            
            return f"Mock documentation for: {query}"
        
        async def get_library_docs(self, library_id: str, topic: str = None) -> str:
            key = f"{library_id}.{topic}" if topic else library_id
            return self.responses.get(key, f"Mock docs for {library_id}")
    
    return MockContext7Client()
```

#### 1.2 Agent-Specific Test Suites
```python
# tests/unit/test_agents/test_planner_comprehensive.py
import pytest
import json
from unittest.mock import patch, AsyncMock

from src.agents.planner import PlannerAgent
from src.utils.types import WorkflowState, SubTask, TaskType
from tests.framework.base import BaseTestCase

class TestPlannerAgentComprehensive(BaseTestCase):
    """Comprehensive test suite for PlannerAgent."""
    
    @pytest.fixture
    def planner_agent(self):
        """Create planner agent with test configuration."""
        config = {
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        return PlannerAgent(config)
    
    @pytest.mark.asyncio
    async def test_simple_geometry_planning(self, planner_agent, mock_openai_client):
        """Test planning for simple geometry creation."""
        with patch.object(planner_agent, 'openai_client', mock_openai_client):
            state = self.create_mock_workflow_state(
                prompt="Create a blue sphere with radius 3"
            )
            
            response = await planner_agent.process(state)
            
            assert response.success
            assert len(response.data) >= 1
            
            # Verify subtask structure
            geometry_task = next(
                (task for task in response.data if task.type == TaskType.GEOMETRY),
                None
            )
            assert geometry_task is not None
            assert "sphere" in geometry_task.description.lower()
    
    @pytest.mark.asyncio
    async def test_complex_scene_planning(self, planner_agent, mock_openai_client):
        """Test planning for complex multi-object scenes."""
        with patch.object(planner_agent, 'openai_client', mock_openai_client):
            state = self.create_mock_workflow_state(
                prompt="Create a medieval castle with towers, walls, and a surrounding moat with realistic materials and dramatic lighting"
            )
            
            response = await planner_agent.process(state)
            
            assert response.success
            assert len(response.data) >= 4  # Multiple components expected
            
            # Verify task diversity
            task_types = [task.type for task in response.data]
            assert TaskType.GEOMETRY in task_types
            assert TaskType.MATERIAL in task_types
            
            # Verify task dependencies
            geometry_tasks = [t for t in response.data if t.type == TaskType.GEOMETRY]
            material_tasks = [t for t in response.data if t.type == TaskType.MATERIAL]
            
            # Materials should depend on geometry
            for material_task in material_tasks:
                assert any(
                    geom_task.id in material_task.dependencies
                    for geom_task in geometry_tasks
                )
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_prompt(self, planner_agent, mock_openai_client):
        """Test error handling for invalid prompts."""
        mock_openai_client.create_completion = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        with patch.object(planner_agent, 'openai_client', mock_openai_client):
            state = self.create_mock_workflow_state(prompt="")
            
            response = await planner_agent.process(state)
            
            assert not response.success
            assert "error" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_planning_with_constraints(self, planner_agent, mock_openai_client):
        """Test planning with specific constraints."""
        with patch.object(planner_agent, 'openai_client', mock_openai_client):
            state = self.create_mock_workflow_state(
                prompt="Create a low-poly game asset car with under 1000 triangles"
            )
            
            response = await planner_agent.process(state)
            
            assert response.success
            
            # Verify constraint awareness in planning
            geometry_task = next(
                (task for task in response.data if task.type == TaskType.GEOMETRY),
                None
            )
            assert geometry_task is not None
            assert any(
                "low-poly" in str(param).lower()
                for param in geometry_task.parameters.values()
            )
    
    @pytest.mark.parametrize("prompt,expected_min_tasks", [
        ("Create a simple cube", 1),
        ("Create a red metallic cube", 2),
        ("Create a house with windows and door", 3),
        ("Create a complete room with furniture and lighting", 5),
    ])
    @pytest.mark.asyncio
    async def test_task_count_scaling(self, planner_agent, mock_openai_client, prompt, expected_min_tasks):
        """Test that task count scales appropriately with complexity."""
        with patch.object(planner_agent, 'openai_client', mock_openai_client):
            state = self.create_mock_workflow_state(prompt=prompt)
            response = await planner_agent.process(state)
            
            assert response.success
            assert len(response.data) >= expected_min_tasks
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, planner_agent, mock_openai_client):
        """Test performance metrics collection."""
        with patch.object(planner_agent, 'openai_client', mock_openai_client):
            state = self.create_mock_workflow_state()
            
            start_time = asyncio.get_event_loop().time()
            response = await planner_agent.process(state)
            end_time = asyncio.get_event_loop().time()
            
            assert response.success
            assert hasattr(response, 'processing_time')
            assert response.processing_time <= (end_time - start_time) + 0.1  # Small tolerance

# tests/unit/test_agents/test_retrieval_comprehensive.py  
class TestRetrievalAgentComprehensive(BaseTestCase):
    """Comprehensive test suite for RetrievalAgent."""
    
    @pytest.fixture
    def retrieval_agent(self, mock_context7_client):
        """Create retrieval agent with mock Context7 client."""
        config = {'context7_client': mock_context7_client}
        agent = RetrievalAgent(config)
        agent.context7_client = mock_context7_client
        return agent
    
    @pytest.mark.asyncio
    async def test_basic_documentation_retrieval(self, retrieval_agent):
        """Test basic documentation retrieval."""
        subtasks = [
            SubTask(
                id="task_001",
                type=TaskType.GEOMETRY,
                description="Create cube geometry",
                parameters={"primitive": "cube"}
            )
        ]
        
        state = self.create_mock_workflow_state(subtasks=subtasks)
        response = await retrieval_agent.process(state)
        
        assert response.success
        assert "bpy.ops.mesh" in response.data
        assert len(retrieval_agent.context7_client.query_history) > 0
    
    @pytest.mark.asyncio
    async def test_multi_task_documentation_aggregation(self, retrieval_agent):
        """Test documentation aggregation for multiple tasks."""
        subtasks = [
            SubTask(
                id="task_001",
                type=TaskType.GEOMETRY,
                description="Create sphere geometry",
                parameters={"primitive": "sphere"}
            ),
            SubTask(
                id="task_002",
                type=TaskType.MATERIAL,
                description="Apply metallic material",
                parameters={"material_type": "metallic"}
            )
        ]
        
        state = self.create_mock_workflow_state(subtasks=subtasks)
        response = await retrieval_agent.process(state)
        
        assert response.success
        # Should contain documentation for both mesh operations and materials
        assert "bpy.ops.mesh" in response.data
        assert "bpy.data.materials" in response.data
        assert len(retrieval_agent.context7_client.query_history) >= 2
    
    @pytest.mark.asyncio
    async def test_context7_error_handling(self, retrieval_agent):
        """Test graceful handling of Context7 errors."""
        # Configure mock to raise exception
        retrieval_agent.context7_client.query_documentation = AsyncMock(
            side_effect=Exception("Context7 connection error")
        )
        
        state = self.create_mock_workflow_state(
            subtasks=[SubTask(
                id="task_001",
                type=TaskType.GEOMETRY,
                description="Create cube",
                parameters={}
            )]
        )
        
        response = await retrieval_agent.process(state)
        
        # Should handle error gracefully with fallback documentation
        assert response.success
        assert "fallback" in response.data.lower() or "basic" in response.data.lower()
```

### Task 2: Integration Testing Suite
**Duration**: 3-4 days

#### 2.1 End-to-End Workflow Tests
```python
# tests/integration/test_complete_workflows.py
import pytest
import asyncio
from pathlib import Path

from src.workflow.enhanced_graph import create_enhanced_workflow
from src.utils.types import WorkflowState
from tests.framework.base import BaseTestCase

class TestCompleteWorkflows(BaseTestCase):
    """End-to-end workflow integration tests."""
    
    @pytest.fixture
    def workflow_with_mocks(self, mock_openai_client, mock_blender_executor, mock_context7_client):
        """Create workflow with all external dependencies mocked."""
        with patch('src.agents.planner.OpenAIClient', return_value=mock_openai_client), \
             patch('src.agents.coding.OpenAIClient', return_value=mock_openai_client), \
             patch('src.agents.critic.OpenAIClient', return_value=mock_openai_client), \
             patch('src.blender.enhanced_executor.EnhancedBlenderExecutor', return_value=mock_blender_executor), \
             patch('src.knowledge.context7.Context7Client', return_value=mock_context7_client):
            
            workflow = create_enhanced_workflow()
            return workflow
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_asset_creation_workflow(self, workflow_with_mocks):
        """Test complete workflow for simple asset creation."""
        initial_state = WorkflowState(
            prompt="Create a red cube with smooth shading"
        )
        
        # Execute complete workflow
        final_state = await workflow_with_mocks.ainvoke(initial_state)
        
        # Verify successful completion
        assert final_state.should_continue == False  # Workflow completed
        assert final_state.error_message == ""
        assert final_state.execution_result is not None
        assert final_state.execution_result.success
        
        # Verify workflow progression
        assert final_state.subtasks is not None
        assert len(final_state.subtasks) > 0
        assert final_state.documentation != ""
        assert final_state.generated_code != ""
        
        # Verify asset metadata creation
        assert final_state.asset_metadata is not None
        assert final_state.asset_metadata.prompt == initial_state.prompt
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_refinement_cycle(self, workflow_with_mocks, mock_openai_client):
        """Test workflow with automatic refinement cycle."""
        # Configure critic to request refinement
        mock_openai_client.response_templates['critic'] = lambda msgs: '''{
            "overall_score": 5.0,
            "needs_refinement": true,
            "critical_issues": ["Poor lighting", "Missing details"],
            "improvement_suggestions": ["Add better lighting", "Increase detail level"],
            "refinement_priority": "high"
        }'''
        
        initial_state = WorkflowState(
            prompt="Create a detailed medieval sword"
        )
        
        final_state = await workflow_with_mocks.ainvoke(initial_state)
        
        # Verify refinement occurred
        assert final_state.refinement_iterations > 0
        assert final_state.refinement_request is not None
        assert "improvement" in final_state.refinement_request.lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, workflow_with_mocks, mock_blender_executor):
        """Test workflow error handling and recovery."""
        # Configure executor to fail initially, then succeed
        call_count = 0
        original_execute = mock_blender_executor.execute_code
        
        async def failing_execute_code(code, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock_blender_executor.set_failure_mode(True)
            else:
                mock_blender_executor.set_failure_mode(False)
            return await original_execute(code, **kwargs)
        
        mock_blender_executor.execute_code = failing_execute_code
        
        initial_state = WorkflowState(
            prompt="Create a complex architectural structure"
        )
        
        final_state = await workflow_with_mocks.ainvoke(initial_state)
        
        # Verify error was handled and recovery occurred
        assert call_count > 1  # Multiple execution attempts
        assert final_state.execution_result.success  # Eventually succeeded
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, workflow_with_mocks):
        """Test multiple workflows running concurrently."""
        prompts = [
            "Create a red cube",
            "Create a blue sphere", 
            "Create a green cylinder"
        ]
        
        # Execute workflows concurrently
        tasks = [
            workflow_with_mocks.ainvoke(WorkflowState(prompt=prompt))
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all workflows completed successfully
        for i, result in enumerate(results):
            assert result.execution_result.success
            assert prompts[i].split()[2] in result.prompt.lower()  # Color preserved

# tests/integration/test_agent_coordination.py
class TestAgentCoordination(BaseTestCase):
    """Test multi-agent coordination and data flow."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_planner_to_retrieval_data_flow(self, mock_openai_client, mock_context7_client):
        """Test data flow from planner to retrieval agent."""
        from src.agents.planner import PlannerAgent
        from src.agents.retrieval import RetrievalAgent
        
        planner = PlannerAgent({'openai_client': mock_openai_client})
        retrieval = RetrievalAgent({'context7_client': mock_context7_client})
        
        # Execute planner
        state = self.create_mock_workflow_state(
            prompt="Create a textured wooden table"
        )
        
        planner_response = await planner.process(state)
        assert planner_response.success
        
        # Update state with planner results
        state.subtasks = planner_response.data
        
        # Execute retrieval
        retrieval_response = await retrieval.process(state)
        assert retrieval_response.success
        
        # Verify appropriate documentation was retrieved
        assert len(mock_context7_client.query_history) > 0
        assert any("mesh" in query.lower() for query in mock_context7_client.query_history)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_critic_verification_coordination(self, mock_openai_client):
        """Test coordination between critic and verification agents."""
        from src.agents.critic import CriticAgent
        from src.agents.verification import VerificationAgent
        
        # Create test asset files
        asset_files = self.create_test_asset_files()
        
        # Create execution result
        execution_result = self.create_mock_execution_result(
            asset_path=str(asset_files['blend']),
            screenshot_path=str(asset_files['screenshot'])
        )
        
        state = self.create_mock_workflow_state(
            execution_result=execution_result
        )
        
        # Execute critic
        critic = CriticAgent({'openai_client': mock_openai_client})
        critic_response = await critic.process(state)
        assert critic_response.success
        
        state.critic_analysis = critic_response.data
        
        # Execute verification
        verification = VerificationAgent({
            'blender_executable': 'blender',
            'quality_thresholds': {'overall_score': 7.0}
        })
        
        with patch.object(verification, '_run_blender_analysis') as mock_analysis:
            mock_analysis.return_value = {
                'geometry_valid': True,
                'material_count': 1,
                'polygon_count': 24  # Cube
            }
            
            verification_response = await verification.process(state)
            assert verification_response.success
            
            # Verify both analyses are considered
            assert state.critic_analysis is not None
            assert verification_response.data is not None
```

### Task 3: Performance Benchmarking System
**Duration**: 2-3 days

#### 3.1 Performance Test Suite
```python
# tests/performance/test_performance_benchmarks.py
import pytest
import time
import asyncio
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

from src.workflow.enhanced_graph import create_enhanced_workflow
from src.utils.types import WorkflowState
from tests.framework.base import BaseTestCase

@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    test_name: str
    execution_time: float
    memory_usage: float
    success_rate: float
    throughput: float
    metadata: Dict[str, Any]

class TestPerformanceBenchmarks(BaseTestCase):
    """Performance benchmarking test suite."""
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    async def run_performance_test(
        self,
        test_func,
        iterations: int = 10,
        max_concurrent: int = 3
    ) -> PerformanceBenchmark:
        """Run performance test with metrics collection."""
        execution_times = []
        memory_usages = []
        success_count = 0
        
        start_memory = self.measure_memory_usage()
        
        # Run iterations
        for batch_start in range(0, iterations, max_concurrent):
            batch_end = min(batch_start + max_concurrent, iterations)
            batch_size = batch_end - batch_start
            
            # Execute batch concurrently
            tasks = []
            for i in range(batch_size):
                task = asyncio.create_task(self._timed_execution(test_func))
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    execution_times.append(float('inf'))
                else:
                    execution_times.append(result['execution_time'])
                    if result['success']:
                        success_count += 1
            
            # Measure memory after batch
            memory_usages.append(self.measure_memory_usage())
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
        
        end_memory = self.measure_memory_usage()
        
        # Calculate metrics
        valid_times = [t for t in execution_times if t != float('inf')]
        avg_execution_time = statistics.mean(valid_times) if valid_times else float('inf')
        success_rate = success_count / iterations
        throughput = success_count / sum(valid_times) if valid_times else 0
        memory_delta = end_memory - start_memory
        
        return PerformanceBenchmark(
            test_name=test_func.__name__,
            execution_time=avg_execution_time,
            memory_usage=memory_delta,
            success_rate=success_rate,
            throughput=throughput,
            metadata={
                'iterations': iterations,
                'execution_times': execution_times,
                'memory_usages': memory_usages,
                'max_concurrent': max_concurrent
            }
        )
    
    async def _timed_execution(self, test_func) -> Dict[str, Any]:
        """Execute test function with timing."""
        start_time = time.time()
        try:
            result = await test_func()
            success = True
        except Exception as e:
            result = str(e)
            success = False
        
        end_time = time.time()
        
        return {
            'execution_time': end_time - start_time,
            'success': success,
            'result': result
        }
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_simple_workflow_performance(self, workflow_with_mocks):
        """Benchmark simple workflow performance."""
        async def simple_workflow():
            state = WorkflowState(prompt="Create a simple cube")
            result = await workflow_with_mocks.ainvoke(state)
            assert result.execution_result.success
            return result
        
        benchmark = await self.run_performance_test(simple_workflow, iterations=20)
        
        # Performance assertions
        assert benchmark.success_rate >= 0.95  # 95% success rate
        assert benchmark.execution_time < 5.0  # Under 5 seconds average
        assert benchmark.memory_usage < 50.0  # Under 50MB memory increase
        assert benchmark.throughput > 0.1  # At least 0.1 workflows/second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_complex_workflow_performance(self, workflow_with_mocks):
        """Benchmark complex workflow performance."""
        async def complex_workflow():
            state = WorkflowState(
                prompt="Create a detailed architectural building with multiple materials, textures, and realistic lighting"
            )
            result = await workflow_with_mocks.ainvoke(state)
            assert result.execution_result.success
            return result
        
        benchmark = await self.run_performance_test(complex_workflow, iterations=10)
        
        # More relaxed performance assertions for complex workflows
        assert benchmark.success_rate >= 0.90  # 90% success rate
        assert benchmark.execution_time < 15.0  # Under 15 seconds average
        assert benchmark.memory_usage < 100.0  # Under 100MB memory increase
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self, workflow_with_mocks):
        """Benchmark concurrent workflow execution."""
        async def concurrent_workflows():
            tasks = []
            for i in range(5):
                state = WorkflowState(prompt=f"Create object {i}")
                task = asyncio.create_task(workflow_with_mocks.ainvoke(state))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            assert all(r.execution_result.success for r in results)
            return results
        
        benchmark = await self.run_performance_test(
            concurrent_workflows, 
            iterations=5, 
            max_concurrent=1  # Each test runs 5 concurrent workflows
        )
        
        assert benchmark.success_rate >= 0.90
        assert benchmark.execution_time < 20.0  # Under 20 seconds for 5 concurrent
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, workflow_with_mocks):
        """Test for memory leaks during extended execution."""
        memory_measurements = []
        
        for i in range(50):  # Run many iterations
            state = WorkflowState(prompt=f"Create test object {i}")
            await workflow_with_mocks.ainvoke(state)
            
            if i % 10 == 0:  # Measure memory every 10 iterations
                memory_measurements.append(self.measure_memory_usage())
        
        # Check for memory growth trend
        if len(memory_measurements) >= 3:
            # Calculate trend - memory should not grow continuously
            memory_growth = memory_measurements[-1] - memory_measurements[0]
            assert memory_growth < 200.0  # Less than 200MB total growth
            
            # Check for linear growth (potential leak)
            growth_rate = memory_growth / len(memory_measurements)
            assert growth_rate < 10.0  # Less than 10MB per measurement

# tests/performance/test_component_benchmarks.py
class TestComponentBenchmarks(BaseTestCase):
    """Performance benchmarks for individual components."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_planner_agent_performance(self, mock_openai_client):
        """Benchmark planner agent performance."""
        from src.agents.planner import PlannerAgent
        
        planner = PlannerAgent({'openai_client': mock_openai_client})
        
        execution_times = []
        
        test_prompts = [
            "Create a simple cube",
            "Create a house with windows and doors",
            "Create a complex mechanical device with moving parts",
            "Design a futuristic cityscape with multiple buildings"
        ]
        
        for prompt in test_prompts:
            state = self.create_mock_workflow_state(prompt=prompt)
            
            start_time = time.time()
            response = await planner.process(state)
            end_time = time.time()
            
            assert response.success
            execution_times.append(end_time - start_time)
        
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)
        
        # Performance assertions
        assert avg_time < 2.0  # Under 2 seconds average
        assert max_time < 5.0  # No single execution over 5 seconds
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_blender_executor_performance(self, mock_blender_executor):
        """Benchmark Blender executor performance."""
        test_codes = [
            "bpy.ops.mesh.primitive_cube_add()",
            """
            import bpy
            for i in range(10):
                bpy.ops.mesh.primitive_cube_add(location=(i, 0, 0))
            """,
            """
            import bpy
            import bmesh
            
            # Create complex mesh
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16)
            mesh = bpy.data.meshes.new("sphere")
            bm.to_mesh(mesh)
            obj = bpy.data.objects.new("sphere_obj", mesh)
            bpy.context.collection.objects.link(obj)
            """
        ]
        
        execution_times = []
        
        for code in test_codes:
            # Configure executor for different complexities
            if "for i in range" in code:
                mock_blender_executor.set_execution_time(3.0)
            elif "bmesh" in code:
                mock_blender_executor.set_execution_time(4.0)
            else:
                mock_blender_executor.set_execution_time(1.0)
            
            start_time = time.time()
            result = await mock_blender_executor.execute_code(code)
            end_time = time.time()
            
            assert result.success
            execution_times.append(end_time - start_time)
        
        # Verify performance scales reasonably with complexity
        assert execution_times[0] < execution_times[1] < execution_times[2]
        assert max(execution_times) < 10.0  # Even complex operations under 10s
```

### Task 4: Quality Evaluation System
**Duration**: 2-3 days

#### 4.1 Asset Quality Metrics
```python
# tests/evaluation/test_quality_metrics.py
import pytest
from pathlib import Path
from typing import Dict, List, Tuple
import json

from src.evaluation.quality_evaluator import QualityEvaluator
from src.evaluation.metrics import (
    PromptAdherenceMetric,
    GeometricQualityMetric,
    MaterialRealismMetric,
    CodeModularityMetric
)
from tests.framework.base import BaseTestCase

class TestQualityEvaluationSystem(BaseTestCase):
    """Test suite for asset quality evaluation."""
    
    @pytest.fixture
    def quality_evaluator(self):
        """Create quality evaluator with test configuration."""
        return QualityEvaluator(
            metrics_config={
                'prompt_adherence': {'weight': 0.3, 'threshold': 7.0},
                'geometric_quality': {'weight': 0.3, 'threshold': 6.0},
                'material_realism': {'weight': 0.2, 'threshold': 6.0},
                'code_modularity': {'weight': 0.2, 'threshold': 5.0}
            }
        )
    
    @pytest.mark.asyncio
    async def test_prompt_adherence_evaluation(self, quality_evaluator):
        """Test prompt adherence metric evaluation."""
        test_cases = [
            {
                'prompt': 'Create a red cube',
                'asset_description': 'A red cubic object with clean geometry',
                'expected_score_range': (8.0, 10.0)
            },
            {
                'prompt': 'Create a blue sphere with metallic material',
                'asset_description': 'A red cube with basic material',
                'expected_score_range': (2.0, 4.0)  # Poor adherence
            },
            {
                'prompt': 'Create a complex architectural building',
                'asset_description': 'A detailed multi-story building with windows and doors',
                'expected_score_range': (7.0, 9.0)
            }
        ]
        
        for case in test_cases:
            asset_files = self.create_test_asset_files()
            
            score = await quality_evaluator.evaluate_prompt_adherence(
                case['prompt'],
                str(asset_files['blend']),
                case['asset_description']
            )
            
            min_score, max_score = case['expected_score_range']
            assert min_score <= score <= max_score, f"Score {score} not in range {case['expected_score_range']}"
    
    @pytest.mark.asyncio
    async def test_geometric_quality_evaluation(self, quality_evaluator):
        """Test geometric quality metric evaluation."""
        asset_files = self.create_test_asset_files()
        
        # Create mock geometry analysis results
        geometry_data = {
            'vertex_count': 24,  # Cube vertices
            'face_count': 12,    # Cube faces
            'topology_quality': 'good',
            'manifold_edges': True,
            'duplicate_vertices': 0,
            'degenerate_faces': 0
        }
        
        with patch.object(quality_evaluator, '_analyze_geometry') as mock_analyze:
            mock_analyze.return_value = geometry_data
            
            score = await quality_evaluator.evaluate_geometric_quality(
                str(asset_files['blend'])
            )
            
            assert 6.0 <= score <= 10.0  # Should be good quality
    
    @pytest.mark.asyncio
    async def test_material_realism_evaluation(self, quality_evaluator):
        """Test material realism metric evaluation."""
        asset_files = self.create_test_asset_files()
        
        material_data = {
            'material_count': 1,
            'has_pbr_materials': True,
            'texture_usage': 'basic',
            'material_complexity': 'medium',
            'realistic_properties': True
        }
        
        with patch.object(quality_evaluator, '_analyze_materials') as mock_analyze:
            mock_analyze.return_value = material_data
            
            score = await quality_evaluator.evaluate_material_realism(
                str(asset_files['blend'])
            )
            
            assert 5.0 <= score <= 9.0
    
    @pytest.mark.asyncio 
    async def test_code_modularity_evaluation(self, quality_evaluator):
        """Test code modularity metric evaluation."""
        test_codes = [
            # Well-structured modular code
            """
            import bpy
            
            def create_cube(size=2.0, location=(0, 0, 0)):
                bpy.ops.mesh.primitive_cube_add(size=size, location=location)
                return bpy.context.active_object
            
            def apply_material(obj, color=(1, 0, 0)):
                material = bpy.data.materials.new(name="TestMaterial")
                material.use_nodes = True
                bsdf = material.node_tree.nodes["Principled BSDF"]
                bsdf.inputs[0].default_value = (*color, 1.0)
                obj.data.materials.append(material)
            
            cube = create_cube()
            apply_material(cube, (0.8, 0.2, 0.2))
            """,
            
            # Poorly structured monolithic code
            """
            import bpy
            bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
            cube = bpy.context.active_object
            material = bpy.data.materials.new(name="TestMaterial")
            material.use_nodes = True
            bsdf = material.node_tree.nodes["Principled BSDF"]
            bsdf.inputs[0].default_value = (0.8, 0.2, 0.2, 1.0)
            cube.data.materials.append(material)
            bpy.ops.transform.resize(value=(1.5, 1.5, 1.5))
            bpy.ops.object.shade_smooth()
            """
        ]
        
        scores = []
        for code in test_codes:
            score = await quality_evaluator.evaluate_code_modularity(code)
            scores.append(score)
        
        # First code (modular) should score higher than second (monolithic)
        assert scores[0] > scores[1]
        assert scores[0] >= 7.0  # Well-structured code
        assert scores[1] <= 5.0  # Poorly structured code
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_evaluation(self, quality_evaluator):
        """Test comprehensive quality evaluation combining all metrics."""
        asset_files = self.create_test_asset_files()
        
        evaluation_data = {
            'prompt': 'Create a red metallic cube with smooth shading',
            'asset_path': str(asset_files['blend']),
            'generated_code': """
                def create_metallic_cube():
                    import bpy
                    
                    # Create cube
                    bpy.ops.mesh.primitive_cube_add(size=2.0)
                    cube = bpy.context.active_object
                    
                    # Apply material
                    material = bpy.data.materials.new(name="MetallicRed")
                    material.use_nodes = True
                    bsdf = material.node_tree.nodes["Principled BSDF"]
                    bsdf.inputs[0].default_value = (0.8, 0.1, 0.1, 1.0)
                    bsdf.inputs[4].default_value = 1.0  # Metallic
                    bsdf.inputs[7].default_value = 0.1  # Roughness
                    
                    cube.data.materials.append(material)
                    bpy.ops.object.shade_smooth()
                
                create_metallic_cube()
            """,
            'asset_metadata': {
                'creation_time': 2.5,
                'file_size_mb': 1.2,
                'export_formats': ['blend', 'obj']
            }
        }
        
        # Mock individual metric evaluations
        with patch.object(quality_evaluator, 'evaluate_prompt_adherence', return_value=8.5), \
             patch.object(quality_evaluator, 'evaluate_geometric_quality', return_value=7.0), \
             patch.object(quality_evaluator, 'evaluate_material_realism', return_value=8.0), \
             patch.object(quality_evaluator, 'evaluate_code_modularity', return_value=7.5):
            
            comprehensive_score = await quality_evaluator.evaluate_comprehensive_quality(
                evaluation_data
            )
            
            assert 7.0 <= comprehensive_score.overall_score <= 9.0
            assert comprehensive_score.meets_quality_threshold
            assert len(comprehensive_score.detailed_metrics) == 4
            assert comprehensive_score.recommendations is not None
    
    @pytest.mark.parametrize("asset_type,expected_metrics", [
        ("architectural", ["geometric_complexity", "structural_integrity", "realistic_proportions"]),
        ("character", ["anatomical_accuracy", "rigging_quality", "animation_readiness"]),
        ("vehicle", ["mechanical_accuracy", "surface_quality", "functional_design"]),
        ("environment", ["scene_composition", "lighting_quality", "atmospheric_effects"])
    ])
    @pytest.mark.asyncio
    async def test_asset_type_specific_evaluation(self, quality_evaluator, asset_type, expected_metrics):
        """Test asset type-specific evaluation metrics."""
        asset_files = self.create_test_asset_files()
        
        evaluation_result = await quality_evaluator.evaluate_by_asset_type(
            asset_type,
            str(asset_files['blend']),
            prompt=f"Create a {asset_type} asset"
        )
        
        assert evaluation_result.asset_type == asset_type
        
        # Verify type-specific metrics are included
        metric_names = list(evaluation_result.detailed_metrics.keys())
        for expected_metric in expected_metrics:
            assert any(expected_metric in metric for metric in metric_names)

# src/evaluation/quality_evaluator.py (New Implementation)
import asyncio
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class QualityEvaluationResult:
    """Comprehensive quality evaluation result."""
    overall_score: float
    detailed_metrics: Dict[str, float]
    meets_quality_threshold: bool
    recommendations: List[str]
    asset_type: Optional[str] = None
    evaluation_metadata: Dict[str, Any] = None

class QualityEvaluator:
    """Comprehensive asset quality evaluator."""
    
    def __init__(self, metrics_config: Dict[str, Any]):
        """Initialize quality evaluator."""
        self.metrics_config = metrics_config
        self.quality_threshold = 7.0
        
        # Initialize metric evaluators
        self.metric_evaluators = {
            'prompt_adherence': PromptAdherenceMetric(),
            'geometric_quality': GeometricQualityMetric(),
            'material_realism': MaterialRealismMetric(),
            'code_modularity': CodeModularityMetric()
        }
    
    async def evaluate_comprehensive_quality(self, evaluation_data: Dict[str, Any]) -> QualityEvaluationResult:
        """Perform comprehensive quality evaluation."""
        detailed_metrics = {}
        
        # Run all metric evaluations
        for metric_name, evaluator in self.metric_evaluators.items():
            try:
                score = await self._run_metric_evaluation(metric_name, evaluator, evaluation_data)
                detailed_metrics[metric_name] = score
            except Exception as e:
                logger.error(f"Metric evaluation failed", metric=metric_name, error=str(e))
                detailed_metrics[metric_name] = 0.0
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(detailed_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detailed_metrics)
        
        return QualityEvaluationResult(
            overall_score=overall_score,
            detailed_metrics=detailed_metrics,
            meets_quality_threshold=overall_score >= self.quality_threshold,
            recommendations=recommendations,
            evaluation_metadata={
                'evaluation_timestamp': asyncio.get_event_loop().time(),
                'metrics_config': self.metrics_config
            }
        )
    
    def _calculate_weighted_score(self, detailed_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, score in detailed_metrics.items():
            weight = self.metrics_config.get(metric_name, {}).get('weight', 0.25)
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, detailed_metrics: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        for metric_name, score in detailed_metrics.items():
            threshold = self.metrics_config.get(metric_name, {}).get('threshold', 6.0)
            if score < threshold:
                recommendations.extend(self._get_metric_recommendations(metric_name, score))
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_metric_recommendations(self, metric_name: str, score: float) -> List[str]:
        """Get recommendations for specific metric."""
        recommendations_map = {
            'prompt_adherence': [
                "Review prompt requirements more carefully",
                "Ensure asset matches specified colors and materials",
                "Verify geometric shapes match prompt description"
            ],
            'geometric_quality': [
                "Improve mesh topology and edge flow",
                "Fix non-manifold geometry issues",
                "Optimize polygon count for better performance"
            ],
            'material_realism': [
                "Add PBR materials for more realistic appearance",
                "Include appropriate texture maps",
                "Adjust material properties for better realism"
            ],
            'code_modularity': [
                "Break large code blocks into smaller functions",
                "Add proper error handling",
                "Improve code documentation and comments"
            ]
        }
        
        return recommendations_map.get(metric_name, ["Consider improvements to this aspect"])
```

### Task 5: Production Readiness Testing
**Duration**: 2-3 days

#### 5.1 Error Recovery and Edge Cases
```python
# tests/production/test_error_recovery.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from src.workflow.enhanced_graph import create_enhanced_workflow
from src.utils.types import WorkflowState
from tests.framework.base import BaseTestCase

class TestErrorRecoveryAndEdgeCases(BaseTestCase):
    """Test error recovery and edge case handling."""
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, mock_openai_client):
        """Test recovery from network failures."""
        # Configure intermittent network failures
        failure_count = 0
        original_create_completion = mock_openai_client.create_completion
        
        async def failing_completion(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 attempts
                raise ConnectionError("Network unavailable")
            return await original_create_completion(*args, **kwargs)
        
        mock_openai_client.create_completion = failing_completion
        
        workflow = create_enhanced_workflow()
        
        with patch('src.agents.planner.OpenAIClient', return_value=mock_openai_client):
            state = WorkflowState(prompt="Create a test object")
            
            # Should eventually succeed after retries
            result = await workflow.ainvoke(state)
            
            assert result.execution_result is not None
            assert failure_count > 2  # Retries occurred
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_malformed_prompt_handling(self, workflow_with_mocks):
        """Test handling of malformed or problematic prompts."""
        problematic_prompts = [
            "",  # Empty prompt
            "a" * 10000,  # Extremely long prompt
            "Create a \x00\x01\x02 invalid character object",  # Invalid characters
            "CREATE AN OBJECT WITH ALL CAPS AND LOTS OF !!!!",  # All caps with symbols
            "创建一个红色立方体",  # Non-English prompt
            "Create an object with import os; os.system('rm -rf /')",  # Potential injection
        ]
        
        for prompt in problematic_prompts:
            state = WorkflowState(prompt=prompt)
            
            try:
                result = await workflow_with_mocks.ainvoke(state)
                
                # Should handle gracefully without crashing
                assert hasattr(result, 'error_message')
                if result.error_message:
                    assert "error" in result.error_message.lower()
                
            except Exception as e:
                pytest.fail(f"Unhandled exception for prompt '{prompt[:50]}...': {e}")
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, workflow_with_mocks, mock_blender_executor):
        """Test handling of resource exhaustion scenarios."""
        # Simulate memory exhaustion
        mock_blender_executor.execute_code = AsyncMock(
            side_effect=MemoryError("Insufficient memory")
        )
        
        state = WorkflowState(
            prompt="Create a extremely complex scene with millions of objects"
        )
        
        result = await workflow_with_mocks.ainvoke(state)
        
        # Should handle memory error gracefully
        assert result.error_message is not None
        assert not result.execution_result or not result.execution_result.success
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_concurrent_workflow_stability(self, workflow_with_mocks):
        """Test system stability under high concurrent load."""
        # Create many concurrent workflows
        concurrent_count = 20
        
        async def run_workflow(workflow_id):
            state = WorkflowState(prompt=f"Create object {workflow_id}")
            return await workflow_with_mocks.ainvoke(state)
        
        # Execute all workflows concurrently
        tasks = [run_workflow(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_count = 0
        exception_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                exception_count += 1
            elif hasattr(result, 'execution_result') and result.execution_result and result.execution_result.success:
                successful_count += 1
        
        # Should handle high load reasonably well
        success_rate = successful_count / concurrent_count
        assert success_rate >= 0.8  # At least 80% success rate
        assert exception_count < concurrent_count * 0.1  # Less than 10% exceptions
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, mock_blender_executor):
        """Test workflow timeout handling."""
        # Configure executor to take very long time
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(30)  # Simulate very slow execution
            return mock_blender_executor.create_mock_execution_result()
        
        mock_blender_executor.execute_code = slow_execution
        
        workflow = create_enhanced_workflow()
        
        state = WorkflowState(prompt="Create a complex object")
        
        # Should timeout appropriately
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(workflow.ainvoke(state), timeout=5.0)
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, workflow_with_mocks, mock_openai_client):
        """Test recovery from partial workflow failures."""
        # Configure critic agent to fail, but other agents to succeed
        call_count = 0
        original_completion = mock_openai_client.create_completion
        
        async def selective_failure(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Make critic calls fail
            if 'critic' in str(messages).lower() or 'visual' in str(messages).lower():
                if call_count <= 3:  # Fail first few critic attempts
                    raise Exception("Critic service unavailable")
            
            return await original_completion(messages, **kwargs)
        
        mock_openai_client.create_completion = selective_failure
        
        state = WorkflowState(prompt="Create a detailed object requiring visual analysis")
        result = await workflow_with_mocks.ainvoke(state)
        
        # Should still produce a result, potentially with degraded quality assessment
        assert result.execution_result is not None
        if not result.execution_result.success:
            assert result.error_message is not None

# tests/production/test_scalability.py
class TestScalabilityAndLimits(BaseTestCase):
    """Test system scalability and operational limits."""
    
    @pytest.mark.production
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_extended_operation_stability(self, workflow_with_mocks):
        """Test stability during extended operation periods."""
        # Run workflows continuously for extended period
        duration_minutes = 5
        end_time = asyncio.get_event_loop().time() + (duration_minutes * 60)
        
        execution_count = 0
        error_count = 0
        
        while asyncio.get_event_loop().time() < end_time:
            try:
                state = WorkflowState(prompt=f"Create object {execution_count}")
                result = await workflow_with_mocks.ainvoke(state)
                
                if result.execution_result and result.execution_result.success:
                    execution_count += 1
                else:
                    error_count += 1
                
            except Exception:
                error_count += 1
            
            # Brief pause between executions
            await asyncio.sleep(1)
        
        # Verify reasonable performance over extended period
        total_operations = execution_count + error_count
        success_rate = execution_count / total_operations if total_operations > 0 else 0
        
        assert success_rate >= 0.85  # 85% success rate over extended period
        assert execution_count >= duration_minutes * 2  # At least 2 successful ops per minute
    
    @pytest.mark.production
    @pytest.mark.asyncio
    async def test_memory_usage_bounds(self, workflow_with_mocks):
        """Test that memory usage stays within reasonable bounds."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute many workflows to test memory accumulation
        for i in range(100):
            state = WorkflowState(prompt=f"Create test object {i}")
            await workflow_with_mocks.ainvoke(state)
            
            # Check memory every 20 iterations
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be bounded
                assert memory_growth < 500  # Less than 500MB growth
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        assert total_growth < 1000  # Less than 1GB total growth
    
    @pytest.mark.production
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    @pytest.mark.asyncio
    async def test_batch_processing_scalability(self, workflow_with_mocks, batch_size):
        """Test scalability with different batch sizes."""
        # Process batches of workflows
        batch_results = []
        
        for batch_num in range(3):  # Process 3 batches
            batch_tasks = []
            
            for i in range(batch_size):
                state = WorkflowState(prompt=f"Batch {batch_num} Object {i}")
                task = asyncio.create_task(workflow_with_mocks.ainvoke(state))
                batch_tasks.append(task)
            
            batch_start_time = asyncio.get_event_loop().time()
            batch_result = await asyncio.gather(*batch_tasks, return_exceptions=True)
            batch_end_time = asyncio.get_event_loop().time()
            
            # Analyze batch results
            successful_in_batch = sum(
                1 for result in batch_result 
                if not isinstance(result, Exception) 
                and hasattr(result, 'execution_result') 
                and result.execution_result 
                and result.execution_result.success
            )
            
            batch_time = batch_end_time - batch_start_time
            throughput = successful_in_batch / batch_time
            
            batch_results.append({
                'batch_size': batch_size,
                'successful_count': successful_in_batch,
                'batch_time': batch_time,
                'throughput': throughput
            })
        
        # Verify reasonable performance scaling
        avg_throughput = sum(b['throughput'] for b in batch_results) / len(batch_results)
        avg_success_rate = sum(b['successful_count'] for b in batch_results) / (len(batch_results) * batch_size)
        
        assert avg_success_rate >= 0.80  # 80% success rate
        assert avg_throughput > 0.1  # At least 0.1 workflows per second
```

### Task 6: Automated Testing Pipeline
**Duration**: 1-2 days

#### 6.1 CI/CD Integration
```python
# .github/workflows/comprehensive_testing.yml
name: Comprehensive Testing Pipeline

on:
  push:
    branches: [ main, 'phase-*/*' ]
  pull_request:
    branches: [ main ]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run ruff linting
      run: ruff check src/ tests/
    
    - name: Run ruff formatting check
      run: ruff format --check src/ tests/
    
    - name: Run MyPy type checking
      run: mypy src/ --strict
    
    - name: Run Bandit security check
      run: bandit -c pyproject.toml -r src/

  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v \
          --cov=src \
          --cov-report=xml \
          --cov-report=html \
          --cov-fail-under=90
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Install Blender
      run: |
        wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
        tar -xf blender-4.0.0-linux-x64.tar.xz
        sudo ln -s $PWD/blender-4.0.0-linux-x64/blender /usr/local/bin/blender
    
    - name: Run integration tests
      env:
        BLENDER_PATH: /usr/local/bin/blender
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/integration/ -v \
          --timeout=300 \
          -m "not slow"

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install psutil
    
    - name: Run performance benchmarks
      run: |
        pytest tests/performance/ -v \
          --benchmark-only \
          --benchmark-save=benchmark_results
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: .benchmarks/

  production-readiness:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run production readiness tests
      run: |
        pytest tests/production/ -v \
          --timeout=600 \
          -m "not slow"
    
    - name: Run quality evaluation tests
      run: |
        pytest tests/evaluation/ -v \
          --timeout=300

  documentation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install mkdocs mkdocs-material mkdocstrings
    
    - name: Build documentation
      run: mkdocs build --strict
    
    - name: Check docstring coverage
      run: |
        interrogate src/ --fail-under=80

# scripts/run_comprehensive_tests.py
#!/usr/bin/env python3
"""Comprehensive test runner for LL3M project."""

import asyncio
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import structlog

logger = structlog.get_logger(__name__)

class ComprehensiveTestRunner:
    """Comprehensive test runner with detailed reporting."""
    
    def __init__(self, verbose: bool = False, output_dir: str = "test_results"):
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results = {}
        self.start_time = time.time()
    
    def run_test_suite(self, suite_name: str, command: List[str]) -> Dict[str, Any]:
        """Run a test suite and capture results."""
        logger.info(f"Running {suite_name} tests", command=' '.join(command))
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            # Save output to files
            output_file = self.output_dir / f"{suite_name}_output.txt"
            error_file = self.output_dir / f"{suite_name}_errors.txt"
            
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            with open(error_file, 'w') as f:
                f.write(result.stderr)
            
            test_result = {
                'suite': suite_name,
                'success': success,
                'duration': duration,
                'return_code': result.returncode,
                'output_file': str(output_file),
                'error_file': str(error_file)
            }
            
            if self.verbose:
                if result.stdout:
                    print(f"\n{suite_name} Output:")
                    print(result.stdout)
                if result.stderr:
                    print(f"\n{suite_name} Errors:")
                    print(result.stderr)
            
            logger.info(
                f"Completed {suite_name}",
                success=success,
                duration=f"{duration:.2f}s"
            )
            
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"{suite_name} timed out")
            return {
                'suite': suite_name,
                'success': False,
                'duration': 600.0,
                'return_code': -1,
                'error': 'Timeout'
            }
        
        except Exception as e:
            logger.error(f"{suite_name} failed with exception", error=str(e))
            return {
                'suite': suite_name,
                'success': False,
                'duration': 0.0,
                'return_code': -1,
                'error': str(e)
            }
    
    def run_all_tests(self, test_categories: List[str] = None) -> Dict[str, Any]:
        """Run all test categories."""
        if test_categories is None:
            test_categories = [
                'code_quality',
                'unit_tests',
                'integration_tests',
                'performance_tests',
                'production_tests'
            ]
        
        test_suites = {
            'code_quality': [
                ['ruff', 'check', 'src/', 'tests/'],
                ['ruff', 'format', '--check', 'src/', 'tests/'],
                ['mypy', 'src/', '--strict']
            ],
            'unit_tests': [
                ['pytest', 'tests/unit/', '-v', '--cov=src', '--cov-report=xml', '--cov-fail-under=90']
            ],
            'integration_tests': [
                ['pytest', 'tests/integration/', '-v', '--timeout=300', '-m', 'not slow']
            ],
            'performance_tests': [
                ['pytest', 'tests/performance/', '-v', '--benchmark-only']
            ],
            'production_tests': [
                ['pytest', 'tests/production/', '-v', '--timeout=600', '-m', 'not slow'],
                ['pytest', 'tests/evaluation/', '-v', '--timeout=300']
            ]
        }
        
        all_results = []
        
        for category in test_categories:
            if category not in test_suites:
                logger.warning(f"Unknown test category: {category}")
                continue
            
            logger.info(f"Starting {category} test category")
            
            for i, command in enumerate(test_suites[category]):
                suite_name = f"{category}_{i}" if len(test_suites[category]) > 1 else category
                result = self.run_test_suite(suite_name, command)
                all_results.append(result)
        
        return self.generate_summary_report(all_results)
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        total_time = time.time() - self.start_time
        
        successful_suites = [r for r in results if r['success']]
        failed_suites = [r for r in results if not r['success']]
        
        summary = {
            'total_suites': len(results),
            'successful_suites': len(successful_suites),
            'failed_suites': len(failed_suites),
            'success_rate': len(successful_suites) / len(results) if results else 0,
            'total_duration': total_time,
            'results': results
        }
        
        # Generate detailed report
        report_file = self.output_dir / "comprehensive_test_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("LL3M Comprehensive Test Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total Test Suites: {summary['total_suites']}\n")
            f.write(f"Successful: {summary['successful_suites']}\n")
            f.write(f"Failed: {summary['failed_suites']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.2%}\n")
            f.write(f"Total Duration: {summary['total_duration']:.2f} seconds\n\n")
            
            if failed_suites:
                f.write("FAILED SUITES:\n")
                f.write("-" * 20 + "\n")
                for result in failed_suites:
                    f.write(f"• {result['suite']}: {result.get('error', 'Test failures')}\n")
                f.write("\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                duration = result['duration']
                f.write(f"{status} {result['suite']} ({duration:.2f}s)\n")
        
        logger.info(
            "Test summary generated",
            report_file=str(report_file),
            success_rate=f"{summary['success_rate']:.2%}"
        )
        
        return summary

def main():
    """Main entry point for comprehensive test runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive LL3M tests')
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=['code_quality', 'unit_tests', 'integration_tests', 'performance_tests', 'production_tests'],
        help='Test categories to run'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    summary = runner.run_all_tests(args.categories)
    
    print(f"\n{'='*50}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Total Suites: {summary['total_suites']}")
    print(f"Successful: {summary['successful_suites']}")
    print(f"Failed: {summary['failed_suites']}")
    print(f"Duration: {summary['total_duration']:.2f}s")
    
    if summary['failed_suites'] > 0:
        print(f"\n❌ {summary['failed_suites']} test suite(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All {summary['successful_suites']} test suites passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Commit remaining Phase 5 work and cleanup", "status": "completed", "activeForm": "Committing remaining Phase 5 work and cleanup"}, {"content": "Create new branch for Phase 6 implementation", "status": "completed", "activeForm": "Creating new branch for Phase 6 implementation"}, {"content": "Create Phase 6 implementation plan using IMPLEMENTATION_PLAN.md and GOOD_PRACTICES.md", "status": "completed", "activeForm": "Creating comprehensive Phase 6 implementation plan"}]