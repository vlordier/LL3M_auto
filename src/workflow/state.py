"""State definitions for LangGraph workflow.

This module provides both legacy monolithic WorkflowState and new domain-specific
state objects for improved modularity and separation of concerns.
"""

# Legacy monolithic state - maintained for backward compatibility
from ..utils.types import WorkflowState

# New domain-specific state objects
from .state_models import (
    CompositeWorkflowState,
    WorkflowAnalysisState,
    WorkflowAssetState,
    WorkflowControlState,
    WorkflowExecutionState,
    WorkflowInputState,
    WorkflowProcessingState,
    WorkflowQualityState,
    WorkflowRefinementState,
)

# Re-export all state types
__all__ = [
    # Legacy
    "WorkflowState",
    # New domain-specific states
    "WorkflowInputState",
    "WorkflowProcessingState", 
    "WorkflowAnalysisState",
    "WorkflowRefinementState",
    "WorkflowControlState",
    "WorkflowAssetState",
    "CompositeWorkflowState",
    # Type aliases
    "WorkflowExecutionState",
    "WorkflowQualityState",
]
