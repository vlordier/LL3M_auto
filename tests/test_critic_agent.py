"""Tests for CriticAgent."""

import json
from unittest.mock import patch

import pytest

from src.agents.critic import CriticAgent
from src.utils.types import AgentType, ExecutionResult, WorkflowState


@pytest.fixture
def critic_config():
    """Critic agent configuration fixture."""
    return {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 2000,
        "max_retries": 3,
        "request_timeout": 30,
    }


@pytest.fixture
def sample_screenshot(tmp_path):
    """Create a sample screenshot file."""
    screenshot_path = tmp_path / "test_screenshot.png"
    # Create a minimal PNG file (1x1 pixel)
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13"
        b"\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```"
        b"\x00\x00\x00\x04\x00\x01\xddCC\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    screenshot_path.write_bytes(png_data)
    return str(screenshot_path)


@pytest.fixture
def workflow_state_with_screenshot(sample_screenshot):
    """WorkflowState with execution result and screenshot."""
    execution_result = ExecutionResult(
        success=True,
        asset_path="/tmp/test_asset.blend",
        screenshot_path=sample_screenshot,
        logs=["Test log"],
        errors=[],
        execution_time=1.0,
    )

    state = WorkflowState(
        prompt="Create a red cube",
        original_prompt="Create a red cube",
        execution_result=execution_result,
        subtasks=[],
    )
    return state


class TestCriticAgent:
    """Test cases for CriticAgent."""

    def test_initialization(self, critic_config):
        """Test agent initialization."""
        agent = CriticAgent(critic_config)

        assert agent.agent_type == AgentType.CRITIC
        assert agent.name == "Visual Quality Critic"
        assert hasattr(agent, "visual_analyzer")
        assert hasattr(agent, "quality_thresholds")

    def test_quality_thresholds(self, critic_config):
        """Test quality thresholds configuration."""
        agent = CriticAgent(critic_config)

        expected_thresholds = {
            "overall_score": 7.0,
            "visual_quality": 6.5,
            "geometry": 7.0,
            "materials": 6.0,
            "lighting": 6.0,
            "composition": 6.0,
            "requirements_match": 8.0,
        }

        assert agent.quality_thresholds == expected_thresholds

    @pytest.mark.asyncio
    async def test_validate_input_valid(
        self, critic_config, workflow_state_with_screenshot
    ):
        """Test input validation with valid state."""
        agent = CriticAgent(critic_config)

        result = await agent.validate_input(workflow_state_with_screenshot)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_input_no_execution_result(self, critic_config):
        """Test input validation without execution result."""
        agent = CriticAgent(critic_config)
        state = WorkflowState(prompt="test")

        result = await agent.validate_input(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_no_screenshot(self, critic_config):
        """Test input validation without screenshot."""
        agent = CriticAgent(critic_config)

        execution_result = ExecutionResult(
            success=True,
            asset_path="/tmp/test.blend",
            screenshot_path=None,
            logs=[],
            errors=[],
            execution_time=1.0,
        )

        state = WorkflowState(prompt="test", execution_result=execution_result)

        result = await agent.validate_input(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_missing_screenshot_file(self, critic_config):
        """Test input validation with missing screenshot file."""
        agent = CriticAgent(critic_config)

        execution_result = ExecutionResult(
            success=True,
            asset_path="/tmp/test.blend",
            screenshot_path="/nonexistent/screenshot.png",
            logs=[],
            errors=[],
            execution_time=1.0,
        )

        state = WorkflowState(prompt="test", execution_result=execution_result)

        result = await agent.validate_input(state)
        assert result is False

    @patch("src.agents.critic.CriticAgent.make_openai_request")
    @pytest.mark.asyncio
    async def test_process_initial_analysis_success(
        self, mock_openai, critic_config, workflow_state_with_screenshot
    ):
        """Test successful initial quality analysis."""
        # Mock OpenAI response
        mock_analysis = {
            "overall_score": 8.5,
            "visual_quality": {"score": 8.0, "notes": "Good visual appeal"},
            "geometry": {"score": 9.0, "notes": "Clean topology"},
            "materials": {"score": 7.0, "notes": "Basic materials"},
            "lighting": {"score": 8.0, "notes": "Good lighting"},
            "composition": {"score": 8.5, "notes": "Nice framing"},
            "requirements_match": {"score": 9.0, "notes": "Matches requirements"},
            "needs_refinement": False,
            "critical_issues": [],
            "improvement_suggestions": [],
            "refinement_priority": "low",
        }

        mock_openai.return_value = json.dumps(mock_analysis)

        agent = CriticAgent(critic_config)
        response = await agent.process(workflow_state_with_screenshot)

        assert response.success is True
        assert response.agent_type == AgentType.CRITIC
        assert response.data == mock_analysis
        assert response.metadata["needs_refinement"] is False
        assert response.metadata["refinement_priority"] == "low"

    @patch("src.agents.critic.CriticAgent.make_openai_request")
    @pytest.mark.asyncio
    async def test_process_needs_refinement(
        self, mock_openai, critic_config, workflow_state_with_screenshot
    ):
        """Test analysis indicating refinement is needed."""
        # Mock analysis with low scores
        mock_analysis = {
            "overall_score": 5.0,
            "visual_quality": {"score": 4.0, "notes": "Poor visual quality"},
            "geometry": {"score": 5.0, "notes": "Rough geometry"},
            "materials": {"score": 3.0, "notes": "Missing materials"},
            "lighting": {"score": 4.0, "notes": "Poor lighting"},
            "composition": {"score": 6.0, "notes": "Adequate framing"},
            "requirements_match": {"score": 7.0, "notes": "Mostly matches"},
            "needs_refinement": True,
            "critical_issues": ["Missing materials", "Poor lighting"],
            "improvement_suggestions": ["Add materials", "Improve lighting"],
            "refinement_priority": "high",
        }

        mock_openai.return_value = json.dumps(mock_analysis)

        agent = CriticAgent(critic_config)
        response = await agent.process(workflow_state_with_screenshot)

        assert response.success is True
        assert response.metadata["needs_refinement"] is True
        assert response.metadata["refinement_priority"] == "high"

    @pytest.mark.asyncio
    async def test_process_invalid_input(self, critic_config):
        """Test process with invalid input."""
        agent = CriticAgent(critic_config)
        state = WorkflowState(prompt="test")

        response = await agent.process(state)

        assert response.success is False
        assert response.agent_type == AgentType.CRITIC
        assert "Invalid input" in response.message

    def test_evaluate_refinement_need_high_overall_score(self, critic_config):
        """Test refinement evaluation with high overall score."""
        agent = CriticAgent(critic_config)

        analysis_result = {
            "overall_score": 9.0,
            "visual_quality": {"score": 8.0},
            "geometry": {"score": 9.0},
            "materials": {"score": 8.0},
            "lighting": {"score": 8.0},
            "composition": {"score": 8.0},
            "requirements_match": {"score": 9.0},
            "critical_issues": [],
        }

        needs_refinement = agent._evaluate_refinement_need(analysis_result)
        assert needs_refinement is False

    def test_evaluate_refinement_need_low_overall_score(self, critic_config):
        """Test refinement evaluation with low overall score."""
        agent = CriticAgent(critic_config)

        analysis_result = {
            "overall_score": 5.0,  # Below threshold of 7.0
            "critical_issues": [],
        }

        needs_refinement = agent._evaluate_refinement_need(analysis_result)
        assert needs_refinement is True

    def test_evaluate_refinement_need_critical_issues(self, critic_config):
        """Test refinement evaluation with critical issues."""
        agent = CriticAgent(critic_config)

        analysis_result = {
            "overall_score": 8.0,  # High score
            "critical_issues": [
                "Missing textures",
                "Poor geometry",
            ],  # But has critical issues
        }

        needs_refinement = agent._evaluate_refinement_need(analysis_result)
        assert needs_refinement is True

    def test_evaluate_refinement_need_explicit_flag(self, critic_config):
        """Test refinement evaluation with explicit needs_refinement flag."""
        agent = CriticAgent(critic_config)

        analysis_result = {
            "needs_refinement": True,
            "overall_score": 8.0,
            "critical_issues": [],
        }

        needs_refinement = agent._evaluate_refinement_need(analysis_result)
        assert needs_refinement is True

    def test_generate_summary_message_initial_analysis(self, critic_config):
        """Test summary message generation for initial analysis."""
        agent = CriticAgent(critic_config)

        analysis_result = {
            "overall_score": 8.5,
            "needs_refinement": False,
            "critical_issues": [],
            "refinement_priority": "low",
        }

        message = agent._generate_summary_message(analysis_result)
        assert "Quality score: 8.5/10" in message
        assert "critical issues" not in message.lower()

    def test_generate_summary_message_with_refinement(self, critic_config):
        """Test summary message generation with refinement needed."""
        agent = CriticAgent(critic_config)

        analysis_result = {
            "overall_score": 6.0,
            "needs_refinement": True,
            "critical_issues": ["Poor lighting", "Missing materials"],
            "refinement_priority": "high",
        }

        message = agent._generate_summary_message(analysis_result)
        assert "Quality score: 6.0/10" in message
        assert "Refinement needed (high priority)" in message
        assert "2 critical issues" in message

    def test_generate_summary_message_comparison(self, critic_config):
        """Test summary message generation for comparison analysis."""
        agent = CriticAgent(critic_config)

        analysis_result = {"improvement_score": 2, "quality_change": "improved"}

        message = agent._generate_summary_message(analysis_result)
        assert "Asset quality improved (+2)" in message
        assert "improved" in message

    @pytest.mark.asyncio
    async def test_encode_image_base64_success(self, critic_config, sample_screenshot):
        """Test successful image encoding to base64."""
        agent = CriticAgent(critic_config)

        result = await agent._encode_image_base64(sample_screenshot)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_encode_image_base64_missing_file(self, critic_config):
        """Test image encoding with missing file."""
        agent = CriticAgent(critic_config)

        result = await agent._encode_image_base64("/nonexistent/file.png")

        assert result is None
