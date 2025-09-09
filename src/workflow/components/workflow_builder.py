"""Modular workflow builder for composing different execution patterns."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ...utils.types import WorkflowState

logger = structlog.get_logger(__name__)


class WorkflowMode(Enum):
    """Different workflow execution modes."""

    FAST = "fast"  # Skip quality assessment, direct execution
    STANDARD = "standard"  # Basic quality checks, minimal refinement
    THOROUGH = "thorough"  # Full quality pipeline, comprehensive refinement
    CUSTOM = "custom"  # User-defined configuration


class QualityLevel(Enum):
    """Quality assessment levels."""

    NONE = "none"  # No quality assessment
    BASIC = "basic"  # Quick validation only
    STANDARD = "standard"  # Critic analysis
    COMPREHENSIVE = "comprehensive"  # Full critic + verification + assessment


class RefinementMode(Enum):
    """Refinement behavior modes."""

    NONE = "none"  # No refinement
    SINGLE = "single"  # One refinement attempt max
    LIMITED = "limited"  # Up to 2 refinements
    ADAPTIVE = "adaptive"  # Dynamic refinement based on quality


@dataclass
class WorkflowConfig:
    """Configuration for workflow behavior."""

    mode: WorkflowMode = WorkflowMode.STANDARD
    quality_level: QualityLevel = QualityLevel.STANDARD
    refinement_mode: RefinementMode = RefinementMode.LIMITED

    # Execution settings
    enable_caching: bool = True
    enable_artifact_management: bool = True
    enable_interruption: bool = True

    # Quality settings
    quality_threshold: float = 2.0
    critical_issues_threshold: int = 5
    skip_perfect_executions: bool = True

    # Refinement settings
    max_refinements: int = 2
    refinement_target: str = "coding"  # Where refinement loops back to
    high_priority_threshold: float = 0.7

    # Performance settings
    recursion_limit: int = 25
    context_compression_threshold: float = 0.85

    # Custom components (for CUSTOM mode)
    custom_nodes: dict[str, Callable] | None = None
    custom_edges: list[dict[str, Any]] | None = None
    custom_conditions: dict[str, Callable] | None = None


class WorkflowBuilder:
    """Builds modular workflows based on configuration."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.graph = StateGraph(WorkflowState)
        self._nodes = {}
        self._edges = []
        self._conditional_edges = []

    def build(self) -> CompiledStateGraph[WorkflowState]:
        """Build the configured workflow."""
        logger.info(f"Building workflow with mode: {self.config.mode.value}")

        # Choose builder strategy based on mode
        if self.config.mode == WorkflowMode.FAST:
            return self._build_fast_workflow()
        elif self.config.mode == WorkflowMode.STANDARD:
            return self._build_standard_workflow()
        elif self.config.mode == WorkflowMode.THOROUGH:
            return self._build_thorough_workflow()
        elif self.config.mode == WorkflowMode.CUSTOM:
            return self._build_custom_workflow()
        else:
            raise ValueError(f"Unknown workflow mode: {self.config.mode}")

    def _build_fast_workflow(self) -> CompiledStateGraph[WorkflowState]:
        """Build fast execution workflow - minimal quality checks."""
        from .streamlined_core import StreamlinedWorkflowBuilder

        builder = StreamlinedWorkflowBuilder(self.config)
        return builder.build_fast_workflow()

    def _build_standard_workflow(self) -> CompiledStateGraph[WorkflowState]:
        """Build standard workflow - balanced quality and speed."""
        from .streamlined_core import StreamlinedWorkflowBuilder

        builder = StreamlinedWorkflowBuilder(self.config)
        return builder.build_standard_workflow()

    def _build_thorough_workflow(self) -> CompiledStateGraph[WorkflowState]:
        """Build thorough workflow - comprehensive quality and refinement."""
        from .streamlined_core import StreamlinedWorkflowBuilder

        builder = StreamlinedWorkflowBuilder(self.config)
        return builder.build_thorough_workflow()

    def _build_custom_workflow(self) -> CompiledStateGraph[WorkflowState]:
        """Build custom workflow from user configuration."""
        if not self.config.custom_nodes:
            raise ValueError("Custom mode requires custom_nodes configuration")

        # Add custom nodes
        for name, node_func in self.config.custom_nodes.items():
            self.graph.add_node(name, node_func)

        # Add custom edges
        if self.config.custom_edges:
            for edge_config in self.config.custom_edges:
                if edge_config.get("conditional"):
                    self.graph.add_conditional_edges(
                        edge_config["from"],
                        edge_config["condition"],
                        edge_config["mapping"],
                    )
                else:
                    self.graph.add_edge(edge_config["from"], edge_config["to"])

        # Set entry point
        entry_point = next(iter(self.config.custom_nodes.keys()))
        self.graph.set_entry_point(entry_point)

        # Compile with configuration
        memory = MemorySaver() if self.config.enable_artifact_management else None
        return self.graph.compile(checkpointer=memory)


def create_workflow(
    config: WorkflowConfig | None = None,
) -> CompiledStateGraph[WorkflowState]:
    """Factory function to create configured workflows."""
    if config is None:
        config = WorkflowConfig()

    builder = WorkflowBuilder(config)
    return builder.build()


# Convenience factory functions
def create_fast_workflow(**kwargs) -> CompiledStateGraph[WorkflowState]:
    """Create a fast execution workflow."""
    config = WorkflowConfig(mode=WorkflowMode.FAST, **kwargs)
    return create_workflow(config)


def create_standard_workflow(**kwargs) -> CompiledStateGraph[WorkflowState]:
    """Create a standard balanced workflow."""
    config = WorkflowConfig(mode=WorkflowMode.STANDARD, **kwargs)
    return create_workflow(config)


def create_thorough_workflow(**kwargs) -> CompiledStateGraph[WorkflowState]:
    """Create a thorough comprehensive workflow."""
    config = WorkflowConfig(mode=WorkflowMode.THOROUGH, **kwargs)
    return create_workflow(config)
