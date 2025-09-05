"""Main entry point for creating LangGraph workflows."""

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ..utils.types import WorkflowState
from .enhanced_graph import (
    create_enhanced_workflow,
    create_simple_workflow,
)


def create_ll3m_workflow(config: dict[str, Any] | None = None) -> CompiledStateGraph[WorkflowState]:
    """Create the main LL3M workflow based on configuration."""
    if config is None:
        config = {}

    workflow_type = config.get("workflow_type", "enhanced")

    if workflow_type in ["fast", "simple"]:
        return create_simple_workflow()

    return create_enhanced_workflow()


__all__ = ["create_ll3m_workflow"]
