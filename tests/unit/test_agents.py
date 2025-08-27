"""Tests for agent implementations."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import EnhancedBaseAgent
from src.agents.planner import PlannerAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.coding import CodingAgent
from src.utils.types import AgentType, TaskType, SubTask, WorkflowState


class TestEnhancedBaseAgent:
    """Tests for EnhancedBaseAgent class."""

    def test_agent_initialization(self, agent_config, mock_openai_client):
        """Test agent can be initialized."""
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_openai_client):
            agent = EnhancedBaseAgent(agent_config)
            assert agent.config == agent_config
            assert agent.client == mock_openai_client
            assert agent.max_retries == 3

    @pytest.mark.asyncio
    async def test_make_openai_request_success(self, agent_config, mock_openai_client):
        """Test successful OpenAI request."""
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_openai_client):
            agent = EnhancedBaseAgent(agent_config)
            
            messages = [{"role": "user", "content": "Test message"}]
            response = await agent.make_openai_request(messages)
            
            assert response == "Test response"
            mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_openai_request_retry(self, agent_config):
        """Test OpenAI request with retries."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            Exception("API Error"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))], 
                     usage=MagicMock(total_tokens=50))
        ]
        
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_client):
            agent = EnhancedBaseAgent(agent_config)
            
            messages = [{"role": "user", "content": "Test message"}]
            response = await agent.make_openai_request(messages)
            
            assert response == "Success"
            assert mock_client.chat.completions.create.call_count == 3

    def test_update_metrics(self, agent_config, mock_openai_client):
        """Test metrics tracking."""
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_openai_client):
            agent = EnhancedBaseAgent(agent_config)
            
            agent._update_metrics(1.5, 100)
            
            assert len(agent.metrics["execution_times"]) == 1
            assert agent.metrics["execution_times"][0] == 1.5
            assert agent.metrics["total_tokens"] == 100
            assert agent.metrics["requests_count"] == 1


class TestPlannerAgent:
    """Tests for PlannerAgent class."""

    @pytest.fixture
    def planner_agent(self, agent_config, mock_openai_client):
        """Create planner agent for testing."""
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_openai_client):
            return PlannerAgent(agent_config)

    def test_planner_initialization(self, planner_agent):
        """Test planner agent initialization."""
        assert planner_agent.agent_type == AgentType.PLANNER
        assert planner_agent.name == "Task Planner"

    @pytest.mark.asyncio
    async def test_process_success(self, planner_agent, sample_workflow_state):
        """Test successful task decomposition."""
        # Mock JSON response
        json_response = {
            "tasks": [
                {
                    "id": "task-1",
                    "type": "geometry",
                    "description": "Create red cube",
                    "priority": 1,
                    "dependencies": [],
                    "parameters": {"shape": "cube", "color": [0.8, 0.2, 0.2]}
                }
            ],
            "reasoning": "Single geometry task for cube creation"
        }
        
        planner_agent.make_openai_request = AsyncMock(
            return_value=json.dumps(json_response)
        )
        
        response = await planner_agent.process(sample_workflow_state)
        
        assert response.success is True
        assert len(response.data) == 1
        assert response.data[0].type == TaskType.GEOMETRY

    @pytest.mark.asyncio
    async def test_process_invalid_json(self, planner_agent, sample_workflow_state):
        """Test handling of invalid JSON response."""
        planner_agent.make_openai_request = AsyncMock(
            return_value="Invalid JSON"
        )
        
        response = await planner_agent.process(sample_workflow_state)
        
        assert response.success is False
        assert "Failed to parse response" in response.message

    @pytest.mark.asyncio
    async def test_validate_input(self, planner_agent):
        """Test input validation."""
        # Valid input
        valid_state = WorkflowState(prompt="Create a cube")
        assert await planner_agent.validate_input(valid_state) is True
        
        # Invalid input - no prompt
        invalid_state = WorkflowState(prompt="")
        assert await planner_agent.validate_input(invalid_state) is False
        
        # Invalid input - prompt too short
        short_state = WorkflowState(prompt="Hi")
        assert await planner_agent.validate_input(short_state) is False

    def test_order_tasks_by_dependencies(self, planner_agent):
        """Test task dependency ordering."""
        tasks = [
            SubTask(id="task-2", type=TaskType.MATERIAL, description="Add material", 
                   priority=2, dependencies=["task-1"]),
            SubTask(id="task-1", type=TaskType.GEOMETRY, description="Create cube", 
                   priority=1, dependencies=[]),
            SubTask(id="task-3", type=TaskType.LIGHTING, description="Add light", 
                   priority=3, dependencies=["task-2"])
        ]
        
        ordered = planner_agent._order_tasks_by_dependencies(tasks)
        
        assert ordered[0].id == "task-1"
        assert ordered[1].id == "task-2"
        assert ordered[2].id == "task-3"


class TestRetrievalAgent:
    """Tests for RetrievalAgent class."""

    @pytest.fixture
    def retrieval_agent(self, agent_config, mock_openai_client, mock_context7_service):
        """Create retrieval agent for testing."""
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_openai_client), \
             patch('src.agents.retrieval.Context7RetrievalService', return_value=mock_context7_service):
            return RetrievalAgent(agent_config)

    def test_retrieval_initialization(self, retrieval_agent):
        """Test retrieval agent initialization."""
        assert retrieval_agent.agent_type == AgentType.RETRIEVAL
        assert retrieval_agent.name == "Documentation Retrieval"

    @pytest.mark.asyncio
    async def test_process_success(self, retrieval_agent, sample_workflow_state, sample_subtask):
        """Test successful documentation retrieval."""
        sample_workflow_state.subtasks = [sample_subtask]
        
        response = await retrieval_agent.process(sample_workflow_state)
        
        assert response.success is True
        assert "Blender Python API Documentation" in response.data
        assert "Sample Blender documentation" in response.data

    def test_extract_topics_from_subtasks(self, retrieval_agent, sample_subtask):
        """Test topic extraction from subtasks."""
        topics = retrieval_agent._extract_topics_from_subtasks([sample_subtask])
        
        assert "geometry" in topics
        assert "cube" in topics

    def test_build_search_queries(self, retrieval_agent, sample_subtask):
        """Test search query building."""
        topics = ["geometry", "cube"]
        queries = retrieval_agent._build_search_queries([sample_subtask], topics)
        
        assert any("geometry" in query for query in queries)
        assert any("cube" in query for query in queries)


class TestCodingAgent:
    """Tests for CodingAgent class."""

    @pytest.fixture
    def coding_agent(self, agent_config, mock_openai_client):
        """Create coding agent for testing."""
        with patch('src.agents.base.AsyncOpenAI', return_value=mock_openai_client):
            return CodingAgent(agent_config)

    def test_coding_initialization(self, coding_agent):
        """Test coding agent initialization."""
        assert coding_agent.agent_type == AgentType.CODING
        assert coding_agent.name == "Code Generator"

    @pytest.mark.asyncio
    async def test_process_success(self, coding_agent, sample_workflow_state, sample_subtask):
        """Test successful code generation."""
        sample_workflow_state.subtasks = [sample_subtask]
        sample_workflow_state.documentation = "Blender docs"
        
        # Mock code generation response
        coding_agent.make_openai_request = AsyncMock(
            return_value="""```python
import bpy
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
```"""
        )
        
        response = await coding_agent.process(sample_workflow_state)
        
        assert response.success is True
        assert "import bpy" in response.data
        assert "primitive_cube_add" in response.data

    def test_clean_generated_code(self, coding_agent):
        """Test code cleaning functionality."""
        raw_code = """```python
import bpy
bpy.ops.mesh.primitive_cube_add()
```
This code creates a cube."""
        
        cleaned = coding_agent._clean_generated_code(raw_code)
        
        assert "```" not in cleaned
        assert "import bpy" in cleaned
        assert "This code creates" not in cleaned

    def test_validate_code_structure(self, coding_agent):
        """Test code validation."""
        # Valid code
        valid_code = "import bpy\nbpy.ops.mesh.primitive_cube_add()"
        result = coding_agent._validate_code_structure(valid_code)
        assert result["valid"] is True
        
        # Invalid code - no bpy import
        invalid_code = "print('hello')"
        result = coding_agent._validate_code_structure(invalid_code)
        assert result["valid"] is False
        assert "Missing bpy import" in result["issues"]

    def test_generate_fallback_code(self, coding_agent, sample_subtask):
        """Test fallback code generation."""
        fallback = coding_agent._generate_fallback_code([sample_subtask])
        
        assert "import bpy" in fallback
        assert "primitive_cube_add" in fallback

    @pytest.mark.asyncio
    async def test_validate_input(self, coding_agent):
        """Test input validation."""
        # Valid input
        valid_state = WorkflowState(
            prompt="test",
            subtasks=[SubTask(id="1", type=TaskType.GEOMETRY, description="test")],
            documentation="docs"
        )
        assert await coding_agent.validate_input(valid_state) is True
        
        # Invalid input - no subtasks
        invalid_state = WorkflowState(prompt="test", subtasks=[])
        assert await coding_agent.validate_input(invalid_state) is False