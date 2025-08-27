"""State definitions for LangGraph workflow."""

from typing import Optional, TypedDict

from ..utils.types import AssetMetadata, ExecutionResult, Issue, SubTask


class LL3MState(TypedDict, total=False):
    """State object for LangGraph workflow.

    This TypedDict defines the structure of data that flows between
    workflow nodes in the LangGraph execution engine.
    """

    # Input phase
    prompt: str
    user_feedback: Optional[str]

    # Planning phase
    subtasks: list[SubTask]

    # Retrieval phase
    documentation: str

    # Coding phase
    generated_code: str

    # Execution phase
    execution_result: Optional[ExecutionResult]
    asset_path: Optional[str]
    screenshot_path: Optional[str]

    # Analysis phase
    detected_issues: list[Issue]

    # Refinement tracking
    refinement_count: int
    max_refinements: int

    # Asset tracking
    asset_metadata: Optional[AssetMetadata]

    # Workflow control
    should_continue: bool
    error_message: Optional[str]
