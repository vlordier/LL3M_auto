"""Main entry point for creating LangGraph workflows."""

from langgraph.graph import StateGraph

from .enhanced_graph import (
    create_comparison_workflow,
    create_enhanced_workflow,
    create_fast_workflow,
)


def create_ll3m_workflow(config: dict | None = None) -> StateGraph:
    """Create the main LL3M workflow based on configuration."""
    if config is None:
        config = {}

    workflow_type = config.get("workflow_type", "enhanced")

    if workflow_type == "fast":
        return create_fast_workflow()
    if workflow_type == "comparison":
        return create_comparison_workflow()

    return create_enhanced_workflow()


__all__ = ["create_ll3m_workflow"]
