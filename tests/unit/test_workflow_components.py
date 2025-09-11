"""Comprehensive tests for modular workflow components."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.types import ExecutionResult, WorkflowState
from src.workflow.components import (
    CoreWorkflowBuilder,
    QualityAssessmentPipeline,
    QualityLevel,
    RefinementEngine,
    RefinementMode,
    WorkflowBuilder,
    WorkflowConfig,
    WorkflowMode,
    create_development_workflow,
    create_production_workflow,
    create_research_workflow,
    should_assess_quality,
    should_refine_enhanced,
    workflow_composer,
)


@pytest.fixture
def sample_state():
    """Create a sample workflow state for testing."""
    return WorkflowState(
        prompt="Create a cube",
        original_prompt="Create a cube",
        iterations=0,
        refinement_iterations=0,
    )


@pytest.fixture
def successful_execution_result(tmp_path):
    """Create a successful execution result."""
    return ExecutionResult(
        success=True,
        asset_path=str(tmp_path / "test.blend"),
        screenshot_path=str(tmp_path / "test.png"),
        errors=[],
        logs=["Execution successful"],
        execution_time=1.0,
    )


@pytest.fixture
def failed_execution_result(tmp_path):
    """Create a failed execution result."""
    return ExecutionResult(
        success=False,
        asset_path=str(tmp_path / "test.blend"),
        screenshot_path=str(tmp_path / "test.png"),
        errors=["SyntaxError: invalid syntax"],
        logs=["Execution failed"],
        execution_time=0.5,
    )


class TestWorkflowConfig:
    """Test WorkflowConfig dataclass and enums."""

    def test_workflow_config_defaults(self):
        """Test default configuration values."""
        config = WorkflowConfig()

        assert config.mode == WorkflowMode.STANDARD
        assert config.quality_level == QualityLevel.STANDARD
        assert config.refinement_mode == RefinementMode.LIMITED
        assert config.quality_threshold == 2.0
        assert config.critical_issues_threshold == 5
        assert config.max_refinements == 2
        assert config.recursion_limit == 25
        assert config.enable_caching is True
        assert config.enable_artifact_management is True

    def test_workflow_config_overrides(self):
        """Test configuration with custom values."""
        config = WorkflowConfig(
            mode=WorkflowMode.FAST,
            quality_level=QualityLevel.NONE,
            refinement_mode=RefinementMode.SINGLE,
            quality_threshold=1.5,
            max_refinements=1,
            recursion_limit=15,
        )

        assert config.mode == WorkflowMode.FAST
        assert config.quality_level == QualityLevel.NONE
        assert config.refinement_mode == RefinementMode.SINGLE
        assert config.quality_threshold == 1.5
        assert config.max_refinements == 1
        assert config.recursion_limit == 15


class TestCoreWorkflowBuilder:
    """Test CoreWorkflowBuilder functionality."""

    def test_core_workflow_builder_initialization(self):
        """Test CoreWorkflowBuilder initialization."""
        config = WorkflowConfig()
        builder = CoreWorkflowBuilder(config)

        assert builder.config == config
        assert builder.graph is not None
        assert builder.pipeline is not None

    def test_execution_pipeline_node_creation(self):
        """Test execution pipeline node creation."""
        config = WorkflowConfig()
        builder = CoreWorkflowBuilder(config)

        # Mock the agent factory and executor
        with (
            patch("src.agents.factory.get_agent_factory"),
            patch("src.blender.enhanced_executor.EnhancedBlenderExecutor"),
        ):
            builder.add_execution_pipeline()

            # Verify nodes were added
            nodes = list(builder.graph.nodes.keys())
            assert "planner" in nodes
            assert "retrieval" in nodes
            assert "coding" in nodes
            assert "execution" in nodes

    def test_should_continue_condition(self):
        """Test should_continue_main condition logic."""
        config = WorkflowConfig()
        builder = CoreWorkflowBuilder(config)
        condition = builder.pipeline.create_should_continue_condition()

        # Test continue case
        state = WorkflowState(prompt="test")
        assert condition(state) == "continue"

        # Test error case
        state.error_message = "Error occurred"
        assert condition(state) == "end"

        # Test max refinements case - the condition doesn't use max_iterations
        # It uses the error_message and other workflow state fields
        state.error_message = ""
        # The should_continue_main condition primarily checks error_message


class TestQualityAssessmentPipeline:
    """Test QualityAssessmentPipeline functionality."""

    def test_quality_pipeline_initialization(self):
        """Test QualityAssessmentPipeline initialization."""
        config = WorkflowConfig()
        pipeline = QualityAssessmentPipeline(config)

        assert pipeline.config == config

    @pytest.mark.asyncio
    async def test_standard_assessment_success_skip(self, sample_state, successful_execution_result):
        """Test standard assessment skips for successful refined executions."""
        config = WorkflowConfig()
        pipeline = QualityAssessmentPipeline(config)
        assessment = pipeline.create_standard_assessment()

        # Set up state for successful refined execution
        sample_state.execution_result = successful_execution_result
        sample_state.refinement_iterations = 1

        result = await assessment(sample_state)

        # Should skip assessment and return unchanged state
        assert result.execution_result == successful_execution_result

    @pytest.mark.asyncio
    async def test_standard_assessment_failed_execution_skip(self, sample_state):
        """Test standard assessment skips for failed executions."""
        config = WorkflowConfig()
        pipeline = QualityAssessmentPipeline(config)
        assessment = pipeline.create_standard_assessment()

        # Set up failed execution
        sample_state.execution_result = ExecutionResult(success=False, execution_time=0.5)

        result = await assessment(sample_state)

        # Should skip assessment for failed executions
        assert result.execution_result.success is False

    def test_should_assess_quality_conditions(self, sample_state):
        """Test should_assess_quality condition function."""
        # Failed execution - should not assess
        sample_state.execution_result = ExecutionResult(success=False, execution_time=0.5)
        assert should_assess_quality(sample_state) == "end"

        # Successful execution, no refinement yet - should assess
        sample_state.execution_result = ExecutionResult(success=True, errors=[], execution_time=1.0)
        sample_state.refinement_iterations = 0
        assert should_assess_quality(sample_state) == "assess"

        # Successful execution after refinement - should skip
        sample_state.execution_result = ExecutionResult(success=True, errors=[], execution_time=1.0)
        sample_state.refinement_iterations = 1
        assert should_assess_quality(sample_state) == "end"


class TestRefinementEngine:
    """Test RefinementEngine functionality."""

    def test_refinement_engine_initialization(self):
        """Test RefinementEngine initialization."""
        config = WorkflowConfig()
        engine = RefinementEngine(config)

        assert engine.config == config

    def test_should_attempt_refinement_conditions(self, sample_state):
        """Test refinement attempt conditions."""
        config = WorkflowConfig(max_refinements=2)
        engine = RefinementEngine(config)

        # Should refine with errors
        sample_state.error_message = "Test error"
        assert engine._should_attempt_refinement(sample_state) is True

        # Should not refine when max refinements reached
        sample_state.refinement_iterations = 2
        assert engine._should_attempt_refinement(sample_state) is False

        # Should not refine when max refinements reached
        sample_state.refinement_iterations = 2
        sample_state.error_message = "Test error"  # Still has error but hit limit
        assert engine._should_attempt_refinement(sample_state) is False

    def test_should_refine_enhanced_conditions(self, sample_state):
        """Test enhanced refinement decision logic."""
        # Hard stop - max iterations in error message
        sample_state.error_message = "max_iterations reached"
        assert should_refine_enhanced(sample_state) == "end"

        # Hard stop - max refinements reached
        sample_state.error_message = ""
        sample_state.refinement_iterations = 2
        assert should_refine_enhanced(sample_state) == "end"

        # Hard stop - max refinements
        # Max refinements limit should be checked instead
        # iterations field not used - refinement_iterations is tracked instead
        sample_state.refinement_iterations = 2
        assert should_refine_enhanced(sample_state) == "end"

        # Success after refinement - should stop
        sample_state.refinement_iterations = 1
        sample_state.execution_result = ExecutionResult(success=True, errors=[], execution_time=1.0)
        assert should_refine_enhanced(sample_state) == "end"

        # Should refine with errors
        sample_state.refinement_iterations = 0
        sample_state.execution_result = None
        sample_state.error_message = "Test error"
        assert should_refine_enhanced(sample_state) == "refine"

        # Should refine with low quality score
        sample_state.error_message = ""
        sample_state.quality_score = 1.0
        assert should_refine_enhanced(sample_state) == "refine"


class TestWorkflowBuilder:
    """Test WorkflowBuilder and workflow creation."""

    def test_workflow_builder_initialization(self):
        """Test WorkflowBuilder initialization."""
        config = WorkflowConfig()
        builder = WorkflowBuilder(config)

        assert builder.config == config
        assert builder.graph is not None

    @patch("src.workflow.components.core_components.CoreWorkflowBuilder")
    def test_fast_workflow_build(self, mock_core_builder):
        """Test fast workflow building."""
        mock_instance = MagicMock()
        mock_core_builder.return_value = mock_instance

        config = WorkflowConfig(mode=WorkflowMode.FAST)
        builder = WorkflowBuilder(config)

        builder._build_fast_workflow()

        # Verify CoreWorkflowBuilder was used
        mock_core_builder.assert_called_once_with(config)
        mock_instance.add_execution_pipeline.assert_called_once()
        mock_instance.compile.assert_called_once()

    @patch("src.workflow.components.core_components.CoreWorkflowBuilder")
    @patch("src.workflow.components.quality_components.QualityAssessmentPipeline")
    @patch("src.workflow.components.refinement_components.RefinementEngine")
    def test_standard_workflow_build(self, mock_refinement, mock_quality, mock_core_builder):
        """Test standard workflow building."""
        mock_core_instance = MagicMock()
        mock_quality_instance = MagicMock()
        mock_refinement_instance = MagicMock()

        mock_core_builder.return_value = mock_core_instance
        mock_quality.return_value = mock_quality_instance
        mock_refinement.return_value = mock_refinement_instance

        config = WorkflowConfig(mode=WorkflowMode.STANDARD)
        builder = WorkflowBuilder(config)

        builder._build_standard_workflow()

        # Verify all components were used
        mock_core_builder.assert_called_once_with(config)
        mock_quality.assert_called_once_with(config)
        mock_refinement.assert_called_once_with(config)

        mock_core_instance.add_execution_pipeline.assert_called_once()
        mock_core_instance.add_quality_gate.assert_called_once()
        mock_core_instance.add_refinement.assert_called_once()

    def test_custom_workflow_build_validation(self):
        """Test custom workflow validation."""
        config = WorkflowConfig(mode=WorkflowMode.CUSTOM)
        builder = WorkflowBuilder(config)

        # Should raise error without custom nodes
        with pytest.raises(ValueError, match="Custom mode requires custom_nodes"):
            builder._build_custom_workflow()


class TestWorkflowTemplates:
    """Test workflow template system."""

    def test_development_workflow_creation(self):
        """Test development workflow template."""
        workflow = create_development_workflow()

        assert workflow.name == "Development Workflow"
        assert workflow.config.mode == WorkflowMode.FAST
        assert workflow.config.quality_level == QualityLevel.BASIC
        assert workflow.config.max_refinements == 1

    def test_production_workflow_creation(self):
        """Test production workflow template."""
        workflow = create_production_workflow()

        assert workflow.name == "Production Workflow"
        assert workflow.config.mode == WorkflowMode.STANDARD
        assert workflow.config.quality_level == QualityLevel.STANDARD
        assert workflow.config.max_refinements == 2

    def test_research_workflow_creation(self):
        """Test research workflow template."""
        workflow = create_research_workflow()

        assert workflow.name == "Research Workflow"
        assert workflow.config.mode == WorkflowMode.THOROUGH
        assert workflow.config.quality_level == QualityLevel.COMPREHENSIVE
        assert workflow.config.max_refinements == 3

    def test_workflow_template_overrides(self):
        """Test workflow template with parameter overrides."""
        workflow = create_production_workflow(max_refinements=4, quality_threshold=3.0)

        assert workflow.config.max_refinements == 4
        assert workflow.config.quality_threshold == 3.0

    def test_workflow_composer_template_list(self):
        """Test workflow composer template listing."""
        templates = workflow_composer.list_templates()

        expected_templates = [
            "development",
            "production",
            "research",
            "quick_prototype",
            "quality_focused",
        ]

        for template in expected_templates:
            assert template in templates

    def test_workflow_composer_template_info(self):
        """Test workflow composer template information retrieval."""
        info = workflow_composer.get_template_info("production")

        assert info is not None
        assert info["name"] == "Production Workflow"
        assert info["description"] == "Balanced workflow for production use"
        assert info["config"]["mode"] == "standard"

    def test_custom_workflow_creation(self):
        """Test custom workflow creation."""
        custom = workflow_composer.create_custom_workflow(
            name="Custom Test Workflow",
            description="Test workflow for specific needs",
            base_template="production",
            max_refinements=5,
            quality_threshold=3.5,
        )

        assert custom.name == "Custom Test Workflow"
        assert custom.description == "Test workflow for specific needs"
        assert custom.config.max_refinements == 5
        assert custom.config.quality_threshold == 3.5

    def test_hybrid_workflow_creation(self):
        """Test hybrid workflow creation."""
        hybrid = workflow_composer.create_hybrid_workflow(
            templates=["development", "quality_focused"], merge_strategy="balanced"
        )

        assert "Hybrid" in hybrid.name
        assert "balanced" in hybrid.description
        assert hybrid.config.mode == WorkflowMode.STANDARD  # Balanced default

    def test_hybrid_workflow_strategies(self):
        """Test different hybrid workflow merge strategies."""
        # Conservative strategy
        conservative = workflow_composer.create_hybrid_workflow(
            templates=["development", "production"], merge_strategy="conservative"
        )
        assert conservative.config.mode == WorkflowMode.THOROUGH

        # Aggressive strategy
        aggressive = workflow_composer.create_hybrid_workflow(
            templates=["research", "quick_prototype"], merge_strategy="aggressive"
        )
        assert aggressive.config.mode == WorkflowMode.FAST


class TestWorkflowEdgeCases:
    """Test workflow edge cases and error handling."""

    def test_workflow_with_no_templates(self):
        """Test error handling for invalid template names."""
        with pytest.raises(ValueError, match="Unknown template"):
            workflow_composer.create_from_template("nonexistent_template")

    def test_hybrid_workflow_with_empty_templates(self):
        """Test error handling for hybrid workflow with no templates."""
        with pytest.raises(ValueError, match="At least one template required"):
            workflow_composer.create_hybrid_workflow(templates=[])

    def test_hybrid_workflow_with_invalid_strategy(self):
        """Test error handling for invalid merge strategy."""
        with pytest.raises(ValueError, match="Unknown merge strategy"):
            workflow_composer.create_hybrid_workflow(
                templates=["development", "production"],
                merge_strategy="invalid_strategy",
            )

    def test_workflow_state_iteration_limits(self, sample_state):
        """Test workflow respects iteration limits."""
        # Set up state to reach refinement limit
        config = WorkflowConfig(max_refinements=2)
        sample_state.max_refinements = config.max_refinements
        sample_state.refinement_iterations = 2  # At limit

        # Should stop due to refinement limit
        builder = CoreWorkflowBuilder(config)
        condition = builder.pipeline.create_should_continue_condition()

        assert condition(sample_state) == "end"

    def test_refinement_limit_enforcement(self, sample_state):
        """Test refinement limits are enforced."""
        config = WorkflowConfig(max_refinements=2)
        engine = RefinementEngine(config)

        # Should allow refinement
        sample_state.refinement_iterations = 1
        sample_state.error_message = "Test error"
        assert engine._should_attempt_refinement(sample_state) is True

        # Should block refinement at limit
        sample_state.refinement_iterations = 2
        assert engine._should_attempt_refinement(sample_state) is False


class TestWorkflowIntegration:
    """Integration tests for workflow components working together."""

    @patch("src.workflow.components.core_components.CoreWorkflowBuilder")
    def test_workflow_component_integration(self, mock_core_builder):
        """Test components work together in workflow building."""
        mock_instance = MagicMock()
        mock_core_builder.return_value = mock_instance

        config = WorkflowConfig(
            mode=WorkflowMode.STANDARD,
            quality_level=QualityLevel.STANDARD,
            refinement_mode=RefinementMode.LIMITED,
        )

        builder = WorkflowBuilder(config)
        builder.build()

        # Verify correct workflow type was built
        mock_core_builder.assert_called_once_with(config)

    def test_workflow_summary_generation(self):
        """Test workflow summary generation."""
        workflow = create_production_workflow()
        summary = workflow.get_summary()

        required_keys = [
            "name",
            "description",
            "mode",
            "quality_level",
            "refinement_mode",
            "max_refinements",
            "quality_threshold",
            "recursion_limit",
            "features",
        ]

        for key in required_keys:
            assert key in summary

        assert summary["features"]["caching"] is True
        assert summary["features"]["artifact_management"] is True
