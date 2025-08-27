"""State definitions for LangGraph workflow.

This module provides consistent state definitions for the workflow.
All state management is now handled through WorkflowState in utils.types.
"""

# Import WorkflowState as the canonical state definition
from ..utils.types import WorkflowState

# Re-export for backward compatibility
__all__ = ["WorkflowState"]
