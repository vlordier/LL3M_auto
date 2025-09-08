"""Streamlined core workflow components with DRY principles."""

from collections.abc import Callable

import structlog

from .graph_builder import GraphBuilder, create_refinement_workflow, create_standard_workflow
from .workflow_builder import WorkflowConfig

logger = structlog.get_logger(__name__)


class StreamlinedWorkflowBuilder:
    """Simplified workflow builder that eliminates duplication and complexity.

    This replaces the complex CoreWorkflowBuilder with a clean, DRY approach.
    """

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.builder = GraphBuilder(config)

    def build_fast_workflow(self):
        """Build a fast workflow with minimal components."""
        logger.info("Building fast workflow")
        return create_standard_workflow(self.config, "minimal")

    def build_standard_workflow(self):
        """Build a standard workflow with full pipeline."""
        logger.info("Building standard workflow")
        return create_standard_workflow(self.config, "basic")

    def build_thorough_workflow(self):
        """Build a thorough workflow with refinement loop."""
        logger.info("Building thorough workflow")
        return create_refinement_workflow(self.config, "basic")

    def build_custom_workflow(self, pipeline_config: list, features: list[str] = None):
        """Build a custom workflow with specified features.

        Args:
            pipeline_config: List of (node_name, agent_type, field_mapping) tuples
            features: Optional features like ['quality_gate', 'refinement_loop']
        """
        logger.info(f"Building custom workflow with features: {features}")

        # Add main pipeline
        self.builder.add_linear_pipeline(pipeline_config)

        # Add optional features
        features = features or []
        execution_node = pipeline_config[-1][0] if pipeline_config else "execution"
        coding_node = next((node[0] for node in pipeline_config if node[1] == "coding"), "coding")

        if "quality_gate" in features:
            self.builder.add_quality_gate(execution_node)

        if "refinement_loop" in features:
            self.builder.add_refinement_loop(execution_node, coding_node)

        return self.builder.compile()


# Simplified factory functions for backward compatibility
def create_planner_node(config: WorkflowConfig) -> Callable:
    """Create planner node using streamlined factory."""
    from .node_factory import NodeFactory
    factory = NodeFactory(config)
    return factory.create_agent_node("planner", {"data": "subtasks"})


def create_retrieval_node(config: WorkflowConfig) -> Callable:
    """Create retrieval node using streamlined factory."""
    from .node_factory import NodeFactory
    factory = NodeFactory(config)
    return factory.create_agent_node("retrieval", {"data": "documentation"})


def create_coding_node(config: WorkflowConfig) -> Callable:
    """Create coding node using streamlined factory."""
    from .node_factory import NodeFactory
    factory = NodeFactory(config)
    return factory.create_agent_node("coding", {"data": "generated_code"})


def create_execution_node(config: WorkflowConfig) -> Callable:
    """Create execution node using streamlined factory."""
    from .node_factory import NodeFactory
    factory = NodeFactory(config)
    return factory.create_execution_node()


def create_continue_condition(config: WorkflowConfig) -> Callable:
    """Create continue condition using streamlined factory."""
    from .node_factory import ConditionFactory
    factory = ConditionFactory(config)
    return factory.create_continue_condition()


# Workflow templates for common patterns
class WorkflowTemplates:
    """Predefined workflow templates for common use cases."""

    @staticmethod
    def development_workflow(config: WorkflowConfig):
        """Fast iteration workflow for development."""
        return create_standard_workflow(config, "minimal")

    @staticmethod
    def production_workflow(config: WorkflowConfig):
        """Full pipeline workflow for production."""
        return create_standard_workflow(config, "basic")

    @staticmethod
    def research_workflow(config: WorkflowConfig):
        """Thorough workflow with refinement for research."""
        return create_refinement_workflow(config, "basic")

    @staticmethod
    def quality_focused_workflow(config: WorkflowConfig):
        """Workflow with comprehensive quality assessment."""
        builder = StreamlinedWorkflowBuilder(config)
        pipeline = [
            ("planner", "planner", {"data": "subtasks"}),
            ("retrieval", "retrieval", {"data": "documentation"}),
            ("coding", "coding", {"data": "generated_code"}),
            ("execution", "execution", {}),
        ]
        return builder.build_custom_workflow(pipeline, ["quality_gate", "refinement_loop"])


# Migration helpers for existing code
def migrate_core_workflow_builder(config: WorkflowConfig) -> StreamlinedWorkflowBuilder:
    """Migration helper for existing CoreWorkflowBuilder usage.

    This provides a drop-in replacement with the same interface but streamlined implementation.
    """
    return StreamlinedWorkflowBuilder(config)


def get_pipeline_summary(workflow_type: str) -> dict:
    """Get summary of what a workflow type includes."""
    summaries = {
        "fast": {
            "nodes": ["coding", "execution"],
            "features": ["quick_quality"],
            "description": "Minimal workflow for rapid iteration"
        },
        "standard": {
            "nodes": ["planner", "retrieval", "coding", "execution"],
            "features": ["quick_quality"],
            "description": "Full pipeline workflow"
        },
        "thorough": {
            "nodes": ["planner", "retrieval", "coding", "execution"],
            "features": ["refinement_loop"],
            "description": "Workflow with error recovery and refinement"
        },
        "quality_focused": {
            "nodes": ["planner", "retrieval", "coding", "execution"],
            "features": ["comprehensive_quality", "refinement_loop"],
            "description": "Workflow optimized for quality assurance"
        }
    }
    return summaries.get(workflow_type, {})
