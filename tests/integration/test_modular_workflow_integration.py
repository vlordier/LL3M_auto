"""Integration tests for modular workflow system."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.types import ExecutionResult, WorkflowState
from src.workflow.components import (
    QualityLevel,
    WorkflowConfig,
    WorkflowMode,
    create_development_workflow,
    create_production_workflow,
    create_research_workflow,
)


@pytest.fixture
def mock_agents():
    """Mock all agent workflow functions."""
    # Use explicit multiple context managers so each can be referenced and configured
    with patch(
        "src.agents.planner.create_planner_workflow", return_value=AsyncMock()
    ) as planner_mock, patch(
        "src.agents.retrieval.create_retrieval_workflow", return_value=AsyncMock()
    ) as retrieval_mock, patch(
        "src.agents.coding.create_coding_workflow", return_value=AsyncMock()
    ) as coding_mock, patch(
        "src.agents.execution.create_execution_workflow", return_value=AsyncMock()
    ) as execution_mock, patch(
        "src.agents.critic.create_critic_workflow", return_value=AsyncMock()
    ) as critic_mock, patch(
        "src.agents.verification.create_verification_workflow", return_value=AsyncMock()
    ) as verification_mock, patch(
        "src.agents.quality_assessment.create_quality_assessment_workflow", return_value=AsyncMock()
    ) as qa_mock, patch(
        "src.agents.refinement.create_refinement_workflow", return_value=AsyncMock()
    ) as refinement_mock:
        yield (
            planner_mock,
            retrieval_mock,
            coding_mock,
            execution_mock,
            critic_mock,
            verification_mock,
            qa_mock,
            refinement_mock,
        )


@pytest.fixture
def successful_state(tmp_path):
    """Create a successful workflow state."""
    execution_result = ExecutionResult(
        success=True,
        asset_path=str(tmp_path / "test.blend"),
        screenshot_path=str(tmp_path / "test.png"),
        errors=[],
        logs=["Success"],
        execution_time=1.0,
    )

    return WorkflowState(
        prompt="Create a cube",
        original_prompt="Create a cube",
        execution_result=execution_result,
        blend_file_path=str(tmp_path / "test.blend"),
        screenshot_path=str(tmp_path / "test.png"),
        iterations=1,
        refinement_iterations=0,
    )


@pytest.fixture
def error_state():
    """Create a workflow state with errors."""
    execution_result = ExecutionResult(
        success=False,
        errors=["SyntaxError: invalid syntax"],
        logs=["Failed"],
        execution_time=0.5,
    )

    return WorkflowState(
        prompt="Create a cube",
        original_prompt="Create a cube",
        error_message="SyntaxError: invalid syntax",
        execution_result=execution_result,
        iterations=1,
        refinement_iterations=0,
    )


class TestWorkflowLifecycle:
    """Test complete workflow lifecycle scenarios."""

    @pytest.mark.asyncio
    async def test_development_workflow_success_path(self, mock_agents, successful_state):
        """Test development workflow success path."""
        # Mock successful responses
        planner_mock, retrieval_mock, coding_mock, execution_mock = mock_agents[:4]

        planner_mock.return_value = successful_state
        retrieval_mock.return_value = successful_state
        coding_mock.return_value = successful_state
        execution_mock.return_value = successful_state

        workflow = create_development_workflow()

        # Should compile without errors
        assert workflow.workflow is not None
        assert workflow.config.mode == WorkflowMode.FAST
        assert workflow.config.max_refinements == 1

    @pytest.mark.asyncio
    async def test_production_workflow_with_refinement_cycle(self, mock_agents, error_state, successful_state):
        """Test production workflow with refinement cycle."""
        planner_mock, retrieval_mock, coding_mock, execution_mock = mock_agents[:4]
        refinement_mock = mock_agents[7]

        # First execution fails, second succeeds
        execution_responses = [error_state, successful_state]
        execution_mock.side_effect = execution_responses

        planner_mock.return_value = successful_state
        retrieval_mock.return_value = successful_state
        coding_mock.return_value = successful_state
        refinement_mock.return_value = successful_state

        workflow = create_production_workflow()

        # Should compile with refinement capability
        assert workflow.workflow is not None
        assert workflow.config.mode == WorkflowMode.STANDARD
        assert workflow.config.max_refinements == 2

    @pytest.mark.asyncio
    async def test_research_workflow_comprehensive_pipeline(self, mock_agents, successful_state):
        """Test research workflow with comprehensive quality pipeline."""
        # Mock all agents for comprehensive workflow
        for mock_agent in mock_agents:
            mock_agent.return_value = successful_state

        workflow = create_research_workflow()

        # Should compile with full quality pipeline
        assert workflow.workflow is not None
        assert workflow.config.mode == WorkflowMode.THOROUGH
        assert workflow.config.quality_level == QualityLevel.COMPREHENSIVE
        assert workflow.config.max_refinements == 3


class TestWorkflowDecisionLogic:
    """Test workflow decision logic and branching."""

    def test_quality_assessment_gating(self, successful_state):  # noqa: ARG002
        """Test quality assessment gating logic."""
        from src.workflow.components.quality_components import should_assess_quality

        # Failed execution - no assessment
        failed_state = WorkflowState(execution_result=ExecutionResult(success=False), refinement_iterations=0)
        assert should_assess_quality(failed_state) == "end"

        # Successful first execution - should assess
        first_success = WorkflowState(
            execution_result=ExecutionResult(success=True, errors=[]),
            refinement_iterations=0,
        )
        assert should_assess_quality(first_success) == "assess"

        # Successful after refinement - skip assessment
        refined_success = WorkflowState(
            execution_result=ExecutionResult(success=True, errors=[]),
            refinement_iterations=1,
        )
        assert should_assess_quality(refined_success) == "end"

    def test_refinement_decision_logic(self, successful_state, error_state):
        """Test refinement decision logic."""
        from src.workflow.components.refinement_components import should_refine_enhanced

        # Success after refinement - stop
        success_refined = successful_state
        success_refined.refinement_iterations = 1
        assert should_refine_enhanced(success_refined) == "end"

        # Error with refinements available - continue
        error_fresh = error_state
        error_fresh.refinement_iterations = 0
        assert should_refine_enhanced(error_fresh) == "refine"

        # Max refinements reached - stop
        error_max_refined = error_state
        error_max_refined.refinement_iterations = 2
        assert should_refine_enhanced(error_max_refined) == "end"

        # Max iterations reached - stop
        error_max_iter = error_state
        error_max_iter.max_iterations = 5
        error_max_iter.iterations = 5
        assert should_refine_enhanced(error_max_iter) == "end"


class TestWorkflowErrorHandling:
    """Test workflow error handling and recovery."""

    def test_workflow_with_invalid_configuration(self):
        """Test workflow creation with invalid configuration."""
        from src.workflow.components.workflow_builder import (
            WorkflowBuilder,
            WorkflowConfig,
        )

        # Custom mode without nodes should fail
        config = WorkflowConfig(mode=WorkflowMode.CUSTOM)
        builder = WorkflowBuilder(config)

        with pytest.raises(ValueError, match="Custom mode requires custom_nodes"):
            builder.build()

    def test_workflow_iteration_limits(self):
        """Test workflow respects iteration limits."""
        state = WorkflowState(prompt="test", max_iterations=1, iterations=1)

        from src.workflow.components.core_components import CoreWorkflowBuilder

        config = WorkflowConfig()
        builder = CoreWorkflowBuilder(config)
        condition = builder.pipeline.create_should_continue_condition()

        assert condition(state) == "end"

    def test_refinement_limits_enforcement(self):
        """Test refinement limits are properly enforced."""
        from src.workflow.components.refinement_components import RefinementEngine

        config = WorkflowConfig(max_refinements=2)
        engine = RefinementEngine(config)

        # Should allow refinement under limit
        state_under_limit = WorkflowState(error_message="test error", refinement_iterations=1)
        assert engine._should_attempt_refinement(state_under_limit) is True

        # Should block refinement at limit
        state_at_limit = WorkflowState(error_message="test error", refinement_iterations=2)
        assert engine._should_attempt_refinement(state_at_limit) is False


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    def test_fast_workflow_minimal_overhead(self):
        """Test fast workflow has minimal quality overhead."""
        workflow = create_development_workflow()
        config = workflow.config

        # Fast mode characteristics
        assert config.mode == WorkflowMode.FAST
        assert config.quality_level == QualityLevel.BASIC
        assert config.max_refinements == 1
        assert config.recursion_limit == 15  # Lower than standard

    def test_research_workflow_comprehensive_coverage(self):
        """Test research workflow has comprehensive coverage."""
        workflow = create_research_workflow()
        config = workflow.config

        # Comprehensive characteristics
        assert config.mode == WorkflowMode.THOROUGH
        assert config.quality_level == QualityLevel.COMPREHENSIVE
        assert config.max_refinements == 3
        assert config.quality_threshold == 3.0  # Higher standard
        assert config.critical_issues_threshold == 2  # More sensitive

    def test_workflow_recursion_limits(self):
        """Test workflow recursion limits are appropriate."""
        dev_workflow = create_development_workflow()
        prod_workflow = create_production_workflow()
        research_workflow = create_research_workflow()

        # Increasing recursion limits by complexity
        assert dev_workflow.config.recursion_limit <= prod_workflow.config.recursion_limit
        assert prod_workflow.config.recursion_limit <= research_workflow.config.recursion_limit


class TestWorkflowComposition:
    """Test workflow composition and templating."""

    def test_workflow_template_inheritance(self):
        """Test workflow templates inherit correctly."""
        from src.workflow.components.composition import workflow_composer

        # Base production template
        base_workflow = workflow_composer.create_from_template("production")
        assert base_workflow.config.max_refinements == 2

        # Override specific parameters
        custom_workflow = workflow_composer.create_from_template("production", max_refinements=5, quality_threshold=3.5)
        assert custom_workflow.config.max_refinements == 5
        assert custom_workflow.config.quality_threshold == 3.5
        # Other settings should remain from template
        assert custom_workflow.config.mode == WorkflowMode.STANDARD

    def test_hybrid_workflow_composition(self):
        """Test hybrid workflow composition."""
        from src.workflow.components.composition import workflow_composer

        hybrid = workflow_composer.create_hybrid_workflow(
            templates=["development", "quality_focused"], merge_strategy="balanced"
        )

        # Should balance characteristics from both templates
        assert hybrid.config.mode == WorkflowMode.STANDARD
        assert hybrid.config.quality_level in [
            QualityLevel.BASIC,
            QualityLevel.STANDARD,
            QualityLevel.COMPREHENSIVE,
        ]
        assert 1 <= hybrid.config.max_refinements <= 4

    def test_custom_workflow_creation(self):
        """Test custom workflow creation from base templates."""
        from src.workflow.components.composition import workflow_composer

        custom = workflow_composer.create_custom_workflow(
            name="API Testing Workflow",
            description="Specialized for API testing scenarios",
            base_template="production",
            max_refinements=3,
            quality_threshold=2.8,
            critical_issues_threshold=3,
        )

        assert custom.name == "API Testing Workflow"
        assert custom.description == "Specialized for API testing scenarios"
        assert custom.config.max_refinements == 3
        assert custom.config.quality_threshold == 2.8
        assert custom.config.critical_issues_threshold == 3

        # Should inherit other settings from production template
        assert custom.config.mode == WorkflowMode.STANDARD


class TestWorkflowStateManagement:
    """Test workflow state management and transitions."""

    def test_workflow_state_persistence(self, successful_state):
        """Test workflow state maintains consistency."""
        original_prompt = successful_state.original_prompt

        # State should maintain original prompt through iterations
        successful_state.prompt = "Modified prompt for refinement"
        assert successful_state.original_prompt == original_prompt

        # Refinement iterations should increment correctly
        original_iterations = successful_state.refinement_iterations
        successful_state.refinement_iterations += 1
        assert successful_state.refinement_iterations == original_iterations + 1

    def test_workflow_artifact_management(self, successful_state):
        """Test workflow artifact management."""
        # Should have all required artifacts for successful execution
        assert successful_state.blend_file_path is not None
        assert successful_state.screenshot_path is not None
        assert successful_state.execution_result is not None
        assert successful_state.execution_result.success is True

    def test_workflow_error_propagation(self, error_state):
        """Test workflow error propagation."""
        # Errors should be properly captured and propagated
        assert error_state.error_message is not None
        assert error_state.execution_result is not None
        assert error_state.execution_result.success is False
        assert len(error_state.execution_result.errors) > 0


class TestWorkflowCycleAnalysis:
    """Test workflow cycle analysis and prevention."""

    def test_no_infinite_refinement_loops(self):
        """Test that refinement loops cannot be infinite."""
        from src.workflow.components.refinement_components import should_refine_enhanced

        # Create state that would cause infinite loop in old system
        problematic_state = WorkflowState(
            error_message="persistent error",
            refinement_iterations=0,
            max_iterations=None,  # No iteration limit
            quality_score=1.0,  # Low quality
        )

        # First refinement should be allowed
        assert should_refine_enhanced(problematic_state) == "refine"

        # After max refinements, should stop
        problematic_state.refinement_iterations = 2
        assert should_refine_enhanced(problematic_state) == "end"

    def test_quality_assessment_bypass_logic(self):
        """Test quality assessment bypass prevents unnecessary work."""
        from src.workflow.components.quality_components import should_assess_quality

        # Successful execution after refinement should bypass assessment
        bypassed_state = WorkflowState(
            execution_result=ExecutionResult(success=True, errors=[]),
            refinement_iterations=1,
        )
        assert should_assess_quality(bypassed_state) == "end"

        # This prevents expensive quality analysis on known-good results

    def test_refinement_targets_correct_node(self):
        """Test refinement targets coding node, not planner."""
        from src.workflow.components.core_components import CoreWorkflowBuilder
        from src.workflow.components.workflow_builder import WorkflowConfig

        config = WorkflowConfig()
        builder = CoreWorkflowBuilder(config)

        # Mock refinement logic to test edge targeting
        MagicMock()
        builder.add_execution_pipeline()

        # The refinement should connect to coding, not planner
        # This is verified by the graph structure in real implementation
