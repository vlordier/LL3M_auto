"""Type definitions for LL3M system."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    """Types of 3D modeling subtasks with implicit complexity hierarchy."""

    GEOMETRY = "geometry"  # Level 1: Basic shapes and structures
    SCENE_SETUP = "scene_setup"  # Level 2: Scene organization
    MATERIAL = "material"  # Level 3: Surface properties
    LIGHTING = "lighting"  # Level 4: Illumination
    ANIMATION = "animation"  # Level 5: Motion and dynamics


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
    USER_PROXY = "user_proxy"
    MCTS_DECOMPOSER = "mcts"


class SubTask(BaseModel):
    """A subtask identified by the planner."""

    id: str = Field(..., description="Unique identifier for the subtask")
    type: TaskType = Field(..., description="Type of the subtask")
    description: str = Field(..., description="Description of what needs to be done")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    dependencies: list[str] = Field(default=[], description="IDs of dependent subtasks")
    parameters: dict[str, Any] = Field(default={}, description="Task-specific parameters")


class Issue(BaseModel):
    """An issue detected in a 3D asset."""

    id: str = Field(..., description="Unique identifier for the issue")
    type: IssueType = Field(..., description="Type of the issue")
    description: str = Field(..., description="Description of the issue")
    severity: int = Field(ge=1, le=5, description="Severity level (1-5)")
    suggested_fix: str = Field(..., description="Suggested fix for the issue")
    code_location: str | None = Field(None, description="Location in code that needs fixing")


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
    subtasks: list[SubTask] = Field(default=[], description="Subtasks used to create asset")
    refinement_count: int = Field(default=0, description="Number of refinements applied")
    quality_score: float | None = Field(None, ge=0, le=1, description="Quality assessment score")


class AgentResponse(BaseModel):
    """Standard response format for agents."""

    agent_type: AgentType = Field(..., description="Type of agent that generated response")
    success: bool = Field(..., description="Whether the operation was successful")
    data: Any = Field(None, description="Response data")
    message: str = Field("", description="Human-readable message")
    execution_time: float = Field(..., description="Time taken to process")
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata")


class WorkflowArtifacts(BaseModel):
    """Complete artifact collection for a generation session."""

    # Primary artifacts
    description_file: str | None = Field(None, description="Path to markdown description file")
    generated_code: str | None = Field(None, description="Generated Blender Python code")
    asset_files: list[str] = Field(default=[], description="Generated 3D asset files (.blend, .obj, etc.)")
    render_files: list[str] = Field(default=[], description="Rendered images and screenshots")
    export_files: dict[str, str] = Field(default={}, description="Format -> file path mapping")

    # Metadata artifacts
    generation_metadata: dict[str, Any] = Field(default={}, description="Generation process metadata")
    quality_analysis: dict[str, Any] = Field(default={}, description="Quality assessment results")
    refinement_history: list[dict[str, Any]] = Field(default=[], description="Refinement iteration history")

    # Resumption data
    resume_prompt: str = Field("", description="Prompt to resume from this state")
    workflow_checkpoint: dict[str, Any] = Field(default={}, description="Workflow state for resumption")

    def get_complete_artifact_summary(self) -> str:
        """Generate complete summary of all artifacts."""
        summary = []

        if self.description_file:
            summary.append(f"📄 Description: {self.description_file}")
        if self.generated_code:
            summary.append("💻 Code: Generated Blender Python script")
        if self.asset_files:
            summary.append(f"🎨 Assets: {', '.join([Path(f).name for f in self.asset_files])}")
        if self.render_files:
            summary.append(f"🖼️ Renders: {', '.join([Path(f).name for f in self.render_files])}")
        if self.export_files:
            formats = ', '.join(self.export_files.keys())
            summary.append(f"📦 Exports: {formats} formats")

        return "\n".join(summary) if summary else "No artifacts generated"


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
    execution_result: ExecutionResult | None = Field(None, description="Execution result")

    # Analysis phase
    detected_issues: list[Issue] = Field(default=[], description="Issues detected by critic")

    # Refinement tracking
    refinement_iterations: int = Field(default=0, description="Number of refinement iterations")
    max_refinements: int = Field(default=3, description="Maximum allowed refinements")
    refinement_request: str = Field("", description="Current refinement request")
    needs_refinement: bool = Field(default=False, description="Whether refinement is needed")
    original_prompt: str = Field("", description="Original user prompt for reference")

    # Asset tracking
    asset_metadata: AssetMetadata | None = Field(None, description="Generated asset metadata")
    artifacts: WorkflowArtifacts = Field(default_factory=WorkflowArtifacts, description="Complete artifact collection")

    # Workflow control
    should_continue: bool = Field(default=True, description="Whether to continue refinement")
    error_message: str | None = Field(None, description="Error message if workflow failed")
    # Additional fields for enhanced workflow
    previous_screenshot_path: str | None = Field(None, description="Path to previous screenshot for comparison")
    critic_analysis: dict[str, Any] | None = Field(None, description="Analysis from the critic agent")
    refinement_priority: float = Field(default=0.0, description="Priority score for refinement")
    verification_result: dict[str, Any] | None = Field(None, description="Result from verification agent")
    verification_score: float | None = Field(None, description="Quality score from verification agent")
    has_critical_issues: bool | None = Field(None, description="Whether critical issues were found during verification")

    # User Proxy Agent fields
    user_prompt: str = Field("", description="Generated prompt for user interaction")
    user_accepted: bool = Field(default=False, description="Whether user accepted the result")
    user_terminated: bool = Field(default=False, description="Whether user terminated the workflow")

    # Session tracking
    session_id: str = Field(default="", description="Unique session identifier")
    output_directory: str = Field(default="", description="Base output directory for this session")
    
    # Artifact management and workflow resumption
    workflow_id: str = Field(default="", description="Unique workflow identifier for artifact management")
    stored_artifacts: list[str] = Field(default=[], description="List of artifact IDs stored by artifact manager")
    checkpoint_history: list[dict[str, Any]] = Field(default=[], description="History of checkpoint saves")
    
    # Enhanced workflow state tracking
    healing_iterations: int = Field(default=0, description="Number of code healing iterations performed")
    total_tokens_used: int = Field(default=0, description="Total tokens consumed in this workflow")
    performance_metrics: dict[str, Any] = Field(default={}, description="Performance and quality metrics")

    def generate_resume_prompt(self) -> str:
        """Generate a prompt that can resume this workflow state."""
        resume_parts = []

        # Original context
        resume_parts.append(f"RESUME_SESSION: {self.session_id}")
        resume_parts.append(f"ORIGINAL_PROMPT: {self.original_prompt or self.prompt}")

        # Current state
        if self.refinement_iterations > 0:
            resume_parts.append(f"REFINEMENT_ITERATION: {self.refinement_iterations}")

        if self.user_feedback:
            resume_parts.append(f"LAST_FEEDBACK: {self.user_feedback}")

        # Quality assessment
        if self.critic_analysis:
            overall_score = self.critic_analysis.get('overall_score', 'N/A')
            resume_parts.append(f"CURRENT_QUALITY: {overall_score}/10")

        # Artifacts status
        artifact_summary = self.artifacts.get_complete_artifact_summary()
        if artifact_summary != "No artifacts generated":
            resume_parts.append(f"ARTIFACTS:\n{artifact_summary}")

        return "\n".join(resume_parts)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
    )

    def update_artifacts(self, **kwargs) -> None:
        """Update artifacts with new data."""
        for key, value in kwargs.items():
            if hasattr(self.artifacts, key):
                setattr(self.artifacts, key, value)

    def save_artifacts_checkpoint(self, output_dir: Path) -> dict[str, Any]:
        """Save complete artifacts and return checkpoint data."""
        checkpoint_data = {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "original_prompt": self.original_prompt,
            "refinement_iterations": self.refinement_iterations,
            "artifacts": self.artifacts.model_dump(),
            "workflow_state": {
                "user_feedback": self.user_feedback,
                "critic_analysis": self.critic_analysis,
                "verification_result": self.verification_result,
                "should_continue": self.should_continue,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save checkpoint file
        checkpoint_file = output_dir / "workflow_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        return checkpoint_data
