"""Streamlined graph builder for workflow components."""

from collections.abc import Callable
from typing import Any

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ...utils.types import WorkflowState
from .node_factory import ConditionFactory, NodeFactory
from .workflow_builder import WorkflowConfig

logger = structlog.get_logger(__name__)


class GraphBuilder:
    """Streamlined graph builder that eliminates duplication."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.graph = StateGraph(WorkflowState)
        self.node_factory = NodeFactory(config)
        self.condition_factory = ConditionFactory(config)

    def add_linear_pipeline(self, nodes: list[tuple[str, str, dict]], entry_point: str = None):
        """Add a linear pipeline of nodes with consistent conditional edges.

        Args:
            nodes: List of (node_name, agent_type, field_mapping) tuples
            entry_point: Optional entry point override
        """
        logger.info(f"Adding linear pipeline: {[node[0] for node in nodes]}")

        # Create continue condition once
        should_continue = self.condition_factory.create_continue_condition()

        # Add all nodes
        for node_name, agent_type, field_mapping in nodes:
            if agent_type == "execution":
                node = self.node_factory.create_execution_node()
            else:
                node = self.node_factory.create_agent_node(agent_type, field_mapping)
            self.graph.add_node(node_name, node)

        # Set entry point
        if entry_point:
            self.graph.set_entry_point(entry_point)
        elif nodes:
            self.graph.set_entry_point(nodes[0][0])

        # Add conditional edges between consecutive nodes
        for i in range(len(nodes) - 1):
            current_node = nodes[i][0]
            next_node = nodes[i + 1][0]

            self.graph.add_conditional_edges(current_node, should_continue, {"continue": next_node, "end": END})

    def add_quality_gate(self, source_node: str, gate_type: str = "quick"):
        """Add quality assessment gate after a node.

        Args:
            source_node: Node to attach quality gate to
            gate_type: Type of quality gate (quick, comprehensive)
        """
        gate_name = f"{gate_type}_quality"

        if gate_type == "quick":
            # Simple quality check
            async def quick_quality_check(state: WorkflowState) -> WorkflowState:
                """Quick quality check with bypass for obvious successes."""
                if (
                    state.execution_result
                    and state.execution_result.success
                    and not state.execution_result.errors
                    and state.blend_file_path
                    and state.screenshot_path
                ):
                    state.quality_score = 4.0
                    state.quality_feedback = "Quick assessment: Execution successful with all outputs"
                # Run standard assessment simulation
                elif state.execution_result and state.execution_result.success:
                    state.quality_score = 3.5 if not state.execution_result.errors else 2.5
                    state.quality_feedback = "Quality assessment completed successfully"
                else:
                    state.quality_score = 1.0
                    state.quality_feedback = "Quality assessment identified issues"
                return state

            self.graph.add_node(gate_name, quick_quality_check)
        else:
            # Comprehensive quality assessment
            from .quality_components import QualityAssessmentPipeline

            pipeline = QualityAssessmentPipeline(self.config)
            self.graph.add_node(gate_name, pipeline.create_comprehensive_assessment())

        # Add conditional edge from source to gate
        should_assess = self.condition_factory.create_quality_condition()
        self.graph.add_conditional_edges(source_node, should_assess, {"assess": gate_name, "end": END})
        self.graph.add_edge(gate_name, END)

    def add_refinement_loop(self, source_node: str, target_node: str, refinement_type: str = "simple"):
        """Add refinement loop from source back to target.

        Args:
            source_node: Node to check for refinement need
            target_node: Node to loop back to
            refinement_type: Type of refinement (simple, adaptive)
        """
        refinement_name = f"{refinement_type}_refinement"

        if refinement_type == "simple":
            # Simple refinement that updates prompt
            async def simple_refinement(state: WorkflowState) -> WorkflowState:
                """Simple refinement by updating prompt with error context."""
                state.refinement_iterations = getattr(state, "refinement_iterations", 0) + 1

                if state.error_message:
                    error_context = (
                        f"\n\nPrevious attempt failed with error: {state.error_message}\nPlease fix this issue."
                    )
                    state.prompt = f"{state.original_prompt or state.prompt}{error_context}"
                    state.error_message = ""

                logger.info(f"Simple refinement applied (iteration {state.refinement_iterations})")
                return state

            self.graph.add_node(refinement_name, simple_refinement)
        else:
            # Adaptive refinement
            from .refinement_components import RefinementEngine

            engine = RefinementEngine(self.config)
            self.graph.add_node(refinement_name, engine.create_adaptive_refinement())

        # Add refinement decision logic
        should_refine = self.condition_factory.create_refinement_condition()

        self.graph.add_conditional_edges(source_node, should_refine, {"refine": refinement_name, "end": END})
        self.graph.add_edge(refinement_name, target_node)

    def add_custom_node(self, name: str, node_func: Callable):
        """Add a custom node function."""
        self.graph.add_node(name, node_func)

    def add_custom_edge(self, source: str, target: str):
        """Add a custom edge between nodes."""
        self.graph.add_edge(source, target)

    def add_custom_conditional_edge(self, source: str, condition: Callable, mapping: dict[str, str]):
        """Add a custom conditional edge."""
        self.graph.add_conditional_edges(source, condition, mapping)

    def compile(self) -> Any:
        """Compile the workflow graph."""
        logger.info("Compiling workflow graph")
        memory = MemorySaver() if self.config.enable_artifact_management else None
        return self.graph.compile(checkpointer=memory)


# Predefined pipeline configurations
PIPELINE_CONFIGS = {
    "basic": [
        ("planner", "planner", {"data": "subtasks"}),
        ("retrieval", "retrieval", {"data": "documentation"}),
        ("coding", "coding", {"data": "generated_code"}),
        ("execution", "execution", {}),
    ],
    "minimal": [
        ("coding", "coding", {"data": "generated_code"}),
        ("execution", "execution", {}),
    ],
}


def create_standard_workflow(config: WorkflowConfig, pipeline: str = "basic") -> Any:
    """Create a standard workflow with predefined pipeline.

    Args:
        config: Workflow configuration
        pipeline: Pipeline type (basic, minimal)
    """
    if pipeline not in PIPELINE_CONFIGS:
        raise ValueError(f"Unknown pipeline: {pipeline}. Available: {list(PIPELINE_CONFIGS.keys())}")

    builder = GraphBuilder(config)
    nodes = PIPELINE_CONFIGS[pipeline]

    # Add main pipeline
    builder.add_linear_pipeline(nodes, entry_point=nodes[0][0])

    # Add quality gate after execution
    execution_node = "execution" if pipeline == "basic" else nodes[-1][0]
    builder.add_quality_gate(execution_node, "quick")

    return builder.compile()


def create_refinement_workflow(config: WorkflowConfig, pipeline: str = "basic") -> Any:
    """Create a workflow with refinement loop.

    Args:
        config: Workflow configuration
        pipeline: Pipeline type (basic, minimal)
    """
    if pipeline not in PIPELINE_CONFIGS:
        raise ValueError(f"Unknown pipeline: {pipeline}. Available: {list(PIPELINE_CONFIGS.keys())}")

    builder = GraphBuilder(config)
    nodes = PIPELINE_CONFIGS[pipeline]

    # Add main pipeline
    builder.add_linear_pipeline(nodes, entry_point=nodes[0][0])

    # Add refinement loop from execution back to coding
    execution_node = "execution" if pipeline == "basic" else nodes[-1][0]
    coding_node = "coding"

    builder.add_refinement_loop(execution_node, coding_node, "simple")

    return builder.compile()
