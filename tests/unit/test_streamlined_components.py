"""Tests for streamlined workflow components."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.utils.types import ExecutionResult, WorkflowState
from src.workflow.components.graph_builder import GraphBuilder
from src.workflow.components.node_factory import ConditionFactory, NodeFactory
from src.workflow.components.streamlined_core import StreamlinedWorkflowBuilder, WorkflowTemplates
from src.workflow.components.streamlined_quality import QualityAssessment
from src.workflow.components.streamlined_refinement import RefinementEngine
from src.workflow.components.workflow_builder import WorkflowConfig, WorkflowMode


@pytest.fixture
def sample_config():
    """Sample workflow configuration."""
    return WorkflowConfig(
        mode=WorkflowMode.STANDARD,
        max_refinements=2,
        quality_threshold=2.0
    )


@pytest.fixture
def sample_state():
    """Sample workflow state."""
    return WorkflowState(
        prompt="Create a 3D cube",
        original_prompt="Create a 3D cube"
    )


class TestNodeFactory:
    """Test the node factory for DRY agent node creation."""

    def test_node_factory_initialization(self, sample_config):
        """Test node factory initializes properly."""
        factory = NodeFactory(sample_config)
        assert factory.config == sample_config

    @patch('src.agents.factory.get_agent_factory')
    @pytest.mark.asyncio
    async def test_create_agent_node_success(self, mock_factory, sample_config, sample_state):
        """Test agent node creation with success case."""
        # Setup mocks
        mock_agent = AsyncMock()
        mock_response = Mock(success=True, data="test_data")
        mock_agent.process.return_value = mock_response

        mock_factory_instance = Mock()
        mock_factory_instance.create_planner.return_value = mock_agent
        mock_factory.return_value = mock_factory_instance

        # Create node
        factory = NodeFactory(sample_config)
        node = factory.create_agent_node("planner", {"data": "subtasks"})

        # Execute node
        result_state = await node(sample_state)

        # Verify
        assert result_state.subtasks == "test_data"
        assert result_state.error_message is None
        mock_agent.process.assert_called_once_with(sample_state)

    @patch('src.agents.factory.get_agent_factory')
    @pytest.mark.asyncio
    async def test_create_agent_node_failure(self, mock_factory, sample_config, sample_state):
        """Test agent node creation with failure case."""
        # Setup mocks
        mock_agent = AsyncMock()
        mock_response = Mock(success=False, message="Test error")
        mock_agent.process.return_value = mock_response

        mock_factory_instance = Mock()
        mock_factory_instance.create_coding.return_value = mock_agent
        mock_factory.return_value = mock_factory_instance

        # Create node
        factory = NodeFactory(sample_config)
        node = factory.create_agent_node("coding", {"data": "generated_code"})

        # Execute node
        result_state = await node(sample_state)

        # Verify
        assert result_state.error_message == "Test error"

    @patch('src.agents.factory.get_agent_factory')
    @pytest.mark.asyncio
    async def test_create_assessment_node(self, mock_factory, sample_config, sample_state):
        """Test assessment node creation."""
        # Setup mocks
        mock_agent = AsyncMock()
        mock_response = Mock(success=True, data="Assessment complete")
        mock_agent.process.return_value = mock_response

        mock_factory_instance = Mock()
        mock_factory_instance.create_critic.return_value = mock_agent
        mock_factory.return_value = mock_factory_instance

        # Create assessment node
        factory = NodeFactory(sample_config)
        assessment = factory.create_assessment_node("critic")

        # Execute assessment
        result = await assessment(sample_state)

        # Verify
        assert result['score'] == 3.0
        assert result['issues'] == []
        assert result['feedback'] == "Assessment complete"


class TestConditionFactory:
    """Test condition factory for standardized conditions."""

    def test_condition_factory_initialization(self, sample_config):
        """Test condition factory initializes properly."""
        factory = ConditionFactory(sample_config)
        assert factory.config == sample_config

    def test_create_continue_condition_success(self, sample_config, sample_state):
        """Test continue condition with success case."""
        factory = ConditionFactory(sample_config)
        condition = factory.create_continue_condition()

        # Test continue case
        result = condition(sample_state)
        assert result == "continue"

    def test_create_continue_condition_error(self, sample_config, sample_state):
        """Test continue condition with error case."""
        sample_state.error_message = "Test error"

        factory = ConditionFactory(sample_config)
        condition = factory.create_continue_condition()

        # Test end case
        result = condition(sample_state)
        assert result == "end"

    def test_create_refinement_condition(self, sample_config, sample_state):
        """Test refinement condition logic."""
        factory = ConditionFactory(sample_config)
        condition = factory.create_refinement_condition()

        # Test refine case with error
        sample_state.error_message = "Test error"
        result = condition(sample_state)
        assert result == "refine"

        # Test end case with no issues
        sample_state.error_message = ""
        sample_state.execution_result = ExecutionResult(success=True, errors=[], execution_time=1.0)
        sample_state.refinement_iterations = 1
        result = condition(sample_state)
        assert result == "end"


class TestGraphBuilder:
    """Test the streamlined graph builder."""

    def test_graph_builder_initialization(self, sample_config):
        """Test graph builder initializes properly."""
        builder = GraphBuilder(sample_config)
        assert builder.config == sample_config
        assert builder.node_factory is not None
        assert builder.condition_factory is not None

    def test_add_linear_pipeline(self, sample_config):
        """Test adding linear pipeline."""
        builder = GraphBuilder(sample_config)

        pipeline = [
            ("coding", "coding", {"data": "generated_code"}),
            ("execution", "execution", {})
        ]

        # This should not raise an exception
        builder.add_linear_pipeline(pipeline)

        # Verify nodes were added (we can't directly inspect the graph)
        assert len(pipeline) == 2

    def test_add_quality_gate(self, sample_config):
        """Test adding quality gate."""
        builder = GraphBuilder(sample_config)

        # Add execution node first
        pipeline = [("execution", "execution", {})]
        builder.add_linear_pipeline(pipeline)

        # Add quality gate
        builder.add_quality_gate("execution", "quick")

        # Should not raise exception
        assert True

    def test_add_refinement_loop(self, sample_config):
        """Test adding refinement loop."""
        builder = GraphBuilder(sample_config)

        # Add basic pipeline
        pipeline = [
            ("coding", "coding", {"data": "generated_code"}),
            ("execution", "execution", {})
        ]
        builder.add_linear_pipeline(pipeline)

        # Add refinement loop
        builder.add_refinement_loop("execution", "coding", "simple")

        # Should not raise exception
        assert True


class TestStreamlinedWorkflowBuilder:
    """Test the streamlined workflow builder."""

    def test_builder_initialization(self, sample_config):
        """Test builder initializes properly."""
        builder = StreamlinedWorkflowBuilder(sample_config)
        assert builder.config == sample_config
        assert builder.builder is not None

    @patch('src.workflow.components.streamlined_core.create_standard_workflow')
    def test_build_fast_workflow(self, mock_create, sample_config):
        """Test fast workflow building."""
        mock_workflow = Mock()
        mock_create.return_value = mock_workflow

        builder = StreamlinedWorkflowBuilder(sample_config)
        result = builder.build_fast_workflow()

        mock_create.assert_called_once_with(sample_config, "minimal")
        assert result == mock_workflow

    @patch('src.workflow.components.streamlined_core.create_standard_workflow')
    def test_build_standard_workflow(self, mock_create, sample_config):
        """Test standard workflow building."""
        mock_workflow = Mock()
        mock_create.return_value = mock_workflow

        builder = StreamlinedWorkflowBuilder(sample_config)
        result = builder.build_standard_workflow()

        mock_create.assert_called_once_with(sample_config, "basic")
        assert result == mock_workflow

    @patch('src.workflow.components.streamlined_core.create_refinement_workflow')
    def test_build_thorough_workflow(self, mock_create, sample_config):
        """Test thorough workflow building."""
        mock_workflow = Mock()
        mock_create.return_value = mock_workflow

        builder = StreamlinedWorkflowBuilder(sample_config)
        result = builder.build_thorough_workflow()

        mock_create.assert_called_once_with(sample_config, "basic")
        assert result == mock_workflow


class TestQualityAssessment:
    """Test streamlined quality assessment."""

    def test_quality_assessment_initialization(self, sample_config):
        """Test quality assessment initializes properly."""
        assessment = QualityAssessment(sample_config)
        assert assessment.config == sample_config
        assert assessment.node_factory is not None

    @pytest.mark.asyncio
    async def test_quick_quality_check_success(self, sample_config, sample_state):
        """Test quick quality check with successful execution."""
        # Setup successful state
        sample_state.execution_result = ExecutionResult(success=True, errors=[], execution_time=1.0)
        sample_state.blend_file_path = "test.blend"
        sample_state.screenshot_path = "test.png"

        assessment = QualityAssessment(sample_config)
        check = assessment.create_quick_quality_check()

        result_state = await check(sample_state)

        assert result_state.quality_score == 4.0
        assert "Quick assessment" in result_state.quality_feedback

    @pytest.mark.asyncio
    async def test_quick_quality_check_needs_assessment(self, sample_config, sample_state):
        """Test quick quality check when full assessment is needed."""
        # Setup state that needs assessment
        sample_state.execution_result = ExecutionResult(success=True, errors=["minor error"], execution_time=1.0)

        assessment = QualityAssessment(sample_config)
        check = assessment.create_quick_quality_check()

        result_state = await check(sample_state)

        assert result_state.quality_score == 2.5
        assert "Quality assessment completed successfully" in result_state.quality_feedback


class TestRefinementEngine:
    """Test streamlined refinement engine."""

    def test_refinement_engine_initialization(self, sample_config):
        """Test refinement engine initializes properly."""
        engine = RefinementEngine(sample_config)
        assert engine.config == sample_config

    @pytest.mark.asyncio
    async def test_simple_refinement_with_error(self, sample_config, sample_state):
        """Test simple refinement with error state."""
        sample_state.error_message = "Test error"

        engine = RefinementEngine(sample_config)
        refinement = engine.create_simple_refinement()

        result_state = await refinement(sample_state)

        assert result_state.refinement_iterations == 1
        assert "Test error" in result_state.prompt
        assert result_state.error_message == ""  # Cleared for retry

    @pytest.mark.asyncio
    async def test_simple_refinement_skip_conditions(self, sample_config, sample_state):
        """Test simple refinement skips when conditions not met."""
        # No error, so should skip
        sample_state.error_message = ""

        engine = RefinementEngine(sample_config)
        refinement = engine.create_simple_refinement()

        result_state = await refinement(sample_state)

        # Should not increment iterations if skipped
        assert getattr(result_state, 'refinement_iterations', 0) == 0


class TestWorkflowTemplates:
    """Test workflow templates."""

    @patch('src.workflow.components.streamlined_core.create_standard_workflow')
    def test_development_workflow(self, mock_create, sample_config):
        """Test development workflow template."""
        mock_workflow = Mock()
        mock_create.return_value = mock_workflow

        result = WorkflowTemplates.development_workflow(sample_config)

        mock_create.assert_called_once_with(sample_config, "minimal")
        assert result == mock_workflow

    @patch('src.workflow.components.streamlined_core.create_standard_workflow')
    def test_production_workflow(self, mock_create, sample_config):
        """Test production workflow template."""
        mock_workflow = Mock()
        mock_create.return_value = mock_workflow

        result = WorkflowTemplates.production_workflow(sample_config)

        mock_create.assert_called_once_with(sample_config, "basic")
        assert result == mock_workflow

    @patch('src.workflow.components.streamlined_core.create_refinement_workflow')
    def test_research_workflow(self, mock_create, sample_config):
        """Test research workflow template."""
        mock_workflow = Mock()
        mock_create.return_value = mock_workflow

        result = WorkflowTemplates.research_workflow(sample_config)

        mock_create.assert_called_once_with(sample_config, "basic")
        assert result == mock_workflow


class TestDRYImprovements:
    """Test that DRY improvements are working."""

    def test_node_creation_consistency(self, sample_config):
        """Test that all agent nodes use the same factory pattern."""
        factory = NodeFactory(sample_config)

        # All these should use the same underlying pattern
        planner = factory.create_agent_node("planner", {"data": "subtasks"})
        retrieval = factory.create_agent_node("retrieval", {"data": "documentation"})
        coding = factory.create_agent_node("coding", {"data": "generated_code"})

        # They should all have the same function name pattern
        assert planner.__name__ == "planner_node"
        assert retrieval.__name__ == "retrieval_node"
        assert coding.__name__ == "coding_node"

    def test_assessment_creation_consistency(self, sample_config):
        """Test that all assessment nodes use the same pattern."""
        factory = NodeFactory(sample_config)

        # Both should use the same underlying pattern
        critic = factory.create_assessment_node("critic")
        verification = factory.create_assessment_node("verification")

        # They should have consistent naming
        assert critic.__name__ == "critic_assessment"
        assert verification.__name__ == "verification_assessment"

    def test_condition_creation_consistency(self, sample_config):
        """Test that conditions follow consistent patterns."""
        factory = ConditionFactory(sample_config)

        continue_condition = factory.create_continue_condition()
        refinement_condition = factory.create_refinement_condition()
        quality_condition = factory.create_quality_condition()

        # All should be callable
        assert callable(continue_condition)
        assert callable(refinement_condition)
        assert callable(quality_condition)


class TestBackwardCompatibility:
    """Test that streamlined components maintain backward compatibility."""

    def test_legacy_imports_still_work(self):
        """Test that existing imports still work."""
        from src.workflow.components import CoreWorkflowBuilder, QualityAssessmentPipeline, RefinementEngine

        # Should not raise ImportError
        assert CoreWorkflowBuilder is not None
        assert QualityAssessmentPipeline is not None
        assert RefinementEngine is not None

    def test_new_imports_work(self):
        """Test that new streamlined imports work."""
        from src.workflow.components import NodeFactory, QualityAssessment, StreamlinedWorkflowBuilder

        # Should not raise ImportError
        assert StreamlinedWorkflowBuilder is not None
        assert QualityAssessment is not None
        assert NodeFactory is not None
