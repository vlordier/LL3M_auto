"""Domain-specific state models for workflow management.

This module splits the monolithic WorkflowState into focused, domain-specific state objects
following single responsibility principle.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.types import AssetMetadata, ExecutionResult, Issue, SubTask, WorkflowState


class WorkflowInputState(BaseModel):
    """State for workflow input and user interaction."""

    prompt: str = Field(default="", description="Original user prompt")
    user_feedback: str | None = Field(None, description="User refinement feedback")
    original_prompt: str = Field("", description="Original user prompt for reference")

    # User Proxy Agent fields
    user_prompt: str = Field("", description="Generated prompt for user interaction")
    user_accepted: bool = Field(default=False, description="Whether user accepted the result")
    user_terminated: bool = Field(default=False, description="Whether user terminated the workflow")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowProcessingState(BaseModel):
    """State for core processing phases (planning, retrieval, coding, execution)."""

    # Planning phase
    subtasks: list[SubTask] = Field(default=[], description="Identified subtasks")

    # Retrieval phase
    documentation: str = Field("", description="Retrieved Blender documentation")

    # Coding phase
    generated_code: str = Field("", description="Generated Blender Python code")

    # Execution phase
    execution_result: ExecutionResult | None = Field(None, description="Execution result")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowAnalysisState(BaseModel):
    """State for analysis and quality assessment."""

    # Analysis phase
    detected_issues: list[Issue] = Field(default=[], description="Issues detected by critic")
    previous_screenshot_path: str | None = Field(None, description="Path to previous screenshot for comparison")
    critic_analysis: dict[str, Any] | None = Field(None, description="Analysis from the critic agent")
    verification_result: dict[str, Any] | None = Field(None, description="Result from verification agent")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowRefinementState(BaseModel):
    """State for refinement tracking and iteration control."""

    refinement_iterations: int = Field(default=0, description="Number of refinement iterations")
    max_refinements: int = Field(default=3, description="Maximum allowed refinements")
    refinement_request: str = Field("", description="Current refinement request")
    needs_refinement: bool = Field(default=False, description="Whether refinement is needed")
    refinement_priority: float = Field(default=0.0, description="Priority score for refinement")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowControlState(BaseModel):
    """State for workflow control and error handling."""

    should_continue: bool = Field(default=True, description="Whether to continue refinement")
    error_message: str | None = Field(None, description="Error message if workflow failed")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowAssetState(BaseModel):
    """State for asset management and tracking."""

    asset_metadata: AssetMetadata | None = Field(None, description="Generated asset metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompositeWorkflowState(BaseModel):
    """Composite state that combines all domain-specific state objects.

    This provides a single interface while maintaining internal separation of concerns.
    Use this when you need access to multiple state domains in a single operation.
    """

    input_state: WorkflowInputState = Field(default_factory=WorkflowInputState)
    processing_state: WorkflowProcessingState = Field(default_factory=WorkflowProcessingState)
    analysis_state: WorkflowAnalysisState = Field(default_factory=WorkflowAnalysisState)
    refinement_state: WorkflowRefinementState = Field(default_factory=WorkflowRefinementState)
    control_state: WorkflowControlState = Field(default_factory=WorkflowControlState)
    asset_state: WorkflowAssetState = Field(default_factory=WorkflowAssetState)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
    )

    @classmethod
    def from_legacy_state(cls, legacy_state: WorkflowState) -> "CompositeWorkflowState":
        """Create a CompositeWorkflowState from a legacy WorkflowState."""
        return cls(
            input_state=WorkflowInputState(
                prompt=legacy_state.prompt,
                user_feedback=legacy_state.user_feedback,
                original_prompt=legacy_state.original_prompt,
                user_prompt=legacy_state.user_prompt,
                user_accepted=legacy_state.user_accepted,
                user_terminated=legacy_state.user_terminated,
            ),
            processing_state=WorkflowProcessingState(
                subtasks=legacy_state.subtasks,
                documentation=legacy_state.documentation,
                generated_code=legacy_state.generated_code,
                execution_result=legacy_state.execution_result,
            ),
            analysis_state=WorkflowAnalysisState(
                detected_issues=legacy_state.detected_issues,
                previous_screenshot_path=legacy_state.previous_screenshot_path,
                critic_analysis=legacy_state.critic_analysis,
                verification_result=legacy_state.verification_result,
            ),
            refinement_state=WorkflowRefinementState(
                refinement_iterations=legacy_state.refinement_iterations,
                max_refinements=legacy_state.max_refinements,
                refinement_request=legacy_state.refinement_request,
                needs_refinement=legacy_state.needs_refinement,
                refinement_priority=legacy_state.refinement_priority,
            ),
            control_state=WorkflowControlState(
                should_continue=legacy_state.should_continue,
                error_message=legacy_state.error_message,
            ),
            asset_state=WorkflowAssetState(
                asset_metadata=legacy_state.asset_metadata,
            ),
        )

    def to_legacy_state(self) -> WorkflowState:
        """Convert back to legacy WorkflowState for backward compatibility."""
        return WorkflowState(
            prompt=self.input_state.prompt,
            user_feedback=self.input_state.user_feedback,
            original_prompt=self.input_state.original_prompt,
            user_prompt=self.input_state.user_prompt,
            user_accepted=self.input_state.user_accepted,
            user_terminated=self.input_state.user_terminated,
            subtasks=self.processing_state.subtasks,
            documentation=self.processing_state.documentation,
            generated_code=self.processing_state.generated_code,
            execution_result=self.processing_state.execution_result,
            detected_issues=self.analysis_state.detected_issues,
            previous_screenshot_path=self.analysis_state.previous_screenshot_path,
            critic_analysis=self.analysis_state.critic_analysis,
            verification_result=self.analysis_state.verification_result,
            refinement_iterations=self.refinement_state.refinement_iterations,
            max_refinements=self.refinement_state.max_refinements,
            refinement_request=self.refinement_state.refinement_request,
            needs_refinement=self.refinement_state.needs_refinement,
            refinement_priority=self.refinement_state.refinement_priority,
            should_continue=self.control_state.should_continue,
            error_message=self.control_state.error_message,
            asset_metadata=self.asset_state.asset_metadata,
        )


# Type aliases for common state combinations
WorkflowExecutionState = WorkflowProcessingState  # For execution-focused operations
WorkflowQualityState = WorkflowAnalysisState  # For quality assessment operations
