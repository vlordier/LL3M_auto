"""Type definitions for LL3M system."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of 3D modeling subtasks."""

    GEOMETRY = "geometry"
    MATERIAL = "material"
    LIGHTING = "lighting"
    ANIMATION = "animation"
    SCENE_SETUP = "scene_setup"


class IssueType(str, Enum):
    """Types of issues that can be detected in 3D assets."""

    GEOMETRY_ERROR = "geometry_error"
    MATERIAL_ISSUE = "material_issue"
    LIGHTING_PROBLEM = "lighting_problem"
    SCALE_ISSUE = "scale_issue"
    POSITIONING_ERROR = "positioning_error"


class AgentType(str, Enum):
    """Types of agents in the system."""

    PLANNER = "planner"
    RETRIEVAL = "retrieval"
    CODING = "coding"
    CRITIC = "critic"
    VERIFICATION = "verification"


class SubTask(BaseModel):
    """A subtask identified by the planner."""

    id: str = Field(..., description="Unique identifier for the subtask")
    type: TaskType = Field(..., description="Type of the subtask")
    description: str = Field(..., description="Description of what needs to be done")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    dependencies: list[str] = Field(default=[], description="IDs of dependent subtasks")
    parameters: dict[str, Any] = Field(
        default={}, description="Task-specific parameters"
    )


class Issue(BaseModel):
    """An issue detected in a 3D asset."""

    id: str = Field(..., description="Unique identifier for the issue")
    type: IssueType = Field(..., description="Type of the issue")
    description: str = Field(..., description="Description of the issue")
    severity: int = Field(ge=1, le=5, description="Severity level (1-5)")
    suggested_fix: str = Field(..., description="Suggested fix for the issue")
    code_location: str | None = Field(
        None, description="Location in code that needs fixing"
    )


class ExecutionResult(BaseModel):
    """Result of executing Blender code."""

    success: bool = Field(..., description="Whether execution was successful")
    asset_path: str | None = Field(None, description="Path to generated asset file")
    screenshot_path: str | None = Field(None, description="Path to screenshot")
    logs: list[str] = Field(default=[], description="Execution logs")
    errors: list[str] = Field(default=[], description="Execution errors")
    execution_time: float = Field(..., description="Execution time in seconds")


class AssetMetadata(BaseModel):
    """Metadata for a generated 3D asset."""

    id: str = Field(..., description="Unique identifier for the asset")
    prompt: str = Field(..., description="Original text prompt")
    creation_time: datetime = Field(default_factory=datetime.now)
    file_path: str = Field(..., description="Path to the asset file")
    screenshot_path: str | None = Field(None, description="Path to screenshot")
    subtasks: list[SubTask] = Field(
        default=[], description="Subtasks used to create asset"
    )
    refinement_count: int = Field(
        default=0, description="Number of refinements applied"
    )
    quality_score: float | None = Field(
        None, ge=0, le=1, description="Quality assessment score"
    )


class AgentResponse(BaseModel):
    """Standard response format for agents."""

    agent_type: AgentType = Field(
        ..., description="Type of agent that generated response"
    )
    success: bool = Field(..., description="Whether the operation was successful")
    data: Any = Field(None, description="Response data")
    message: str = Field("", description="Human-readable message")
    execution_time: float = Field(..., description="Time taken to process")
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata")


class WorkflowState(BaseModel):
    """State object for LangGraph workflow."""

    # Input
    prompt: str = Field(..., description="Original user prompt")
    user_feedback: str | None = Field(None, description="User refinement feedback")

    # Planning phase
    subtasks: list[SubTask] = Field(default=[], description="Identified subtasks")

    # Retrieval phase
    documentation: str = Field("", description="Retrieved Blender documentation")

    # Coding phase
    generated_code: str = Field("", description="Generated Blender Python code")

    # Execution phase
    execution_result: ExecutionResult | None = Field(
        None, description="Execution result"
    )

    # Analysis phase
    detected_issues: list[Issue] = Field(
        default=[], description="Issues detected by critic"
    )

    # Refinement tracking
    refinement_count: int = Field(
        default=0, description="Number of refinement iterations"
    )
    max_refinements: int = Field(default=3, description="Maximum allowed refinements")

    # Asset tracking
    asset_metadata: AssetMetadata | None = Field(
        None, description="Generated asset metadata"
    )

    # Workflow control
    should_continue: bool = Field(
        default=True, description="Whether to continue refinement"
    )
    error_message: str | None = Field(
        None, description="Error message if workflow failed"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
