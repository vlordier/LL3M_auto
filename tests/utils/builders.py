"""Test builders for creating common test scenarios.

This module provides builder classes that make it easy to create test data
with sensible defaults while allowing customization of specific fields.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from src.api.models import GenerateAssetRequest
from src.services.batch_processor import BatchJob, BatchJobItem, BatchPriority, BatchStatus
from src.utils.types import (
    AgentResponse,
    AgentType,
    ExecutionResult,
    Issue,
    IssueType,
    SubTask,
    TaskType,
    WorkflowState,
)
from src.workflow.state_models import (
    CompositeWorkflowState,
    WorkflowAnalysisState,
    WorkflowAssetState,
    WorkflowControlState,
    WorkflowInputState,
    WorkflowProcessingState,
    WorkflowRefinementState,
)


class WorkflowStateBuilder:
    """Builder for creating WorkflowState instances with sensible defaults."""

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._prompt = "Create a red cube"
        self._user_feedback = None
        self._subtasks: list[SubTask] = []
        self._documentation = ""
        self._generated_code = ""
        self._execution_result = None
        self._detected_issues: list[Issue] = []
        self._refinement_iterations = 0
        self._max_refinements = 3
        self._needs_refinement = False
        self._should_continue = True
        self._error_message = None
        self._asset_metadata = None
        self._user_accepted = False
        self._user_terminated = False

    def with_prompt(self, prompt: str) -> "WorkflowStateBuilder":
        """Set the user prompt."""
        self._prompt = prompt
        return self

    def with_user_feedback(self, feedback: str) -> "WorkflowStateBuilder":
        """Set user feedback."""
        self._user_feedback = feedback
        return self

    def with_successful_execution(self, asset_path: str = "/tmp/test_asset.blend") -> "WorkflowStateBuilder":
        """Configure state with successful execution result."""
        self._execution_result = ExecutionResult(
            success=True,
            asset_path=asset_path,
            screenshot_path=f"{asset_path}_screenshot.png",
            logs=["Execution completed successfully"],
            errors=[],
            execution_time=2.5,
        )
        self._generated_code = "bpy.ops.mesh.primitive_cube_add()"
        return self

    def with_failed_execution(self, error_message: str = "Execution failed") -> "WorkflowStateBuilder":
        """Configure state with failed execution result."""
        self._execution_result = ExecutionResult(
            success=False,
            asset_path=None,
            screenshot_path=None,
            logs=["Starting execution...", "Error occurred"],
            errors=[error_message],
            execution_time=1.0,
        )
        self._error_message = error_message
        self._should_continue = False
        return self

    def with_subtasks(self, count: int = 3) -> "WorkflowStateBuilder":
        """Add subtasks to the state."""
        self._subtasks = [
            SubTask(
                id=f"task_{i}",
                type=TaskType.GEOMETRY,
                description=f"Create geometry task {i}",
                priority=i,
            )
            for i in range(1, count + 1)
        ]
        return self

    def with_detected_issues(self, count: int = 2) -> "WorkflowStateBuilder":
        """Add detected issues to the state."""
        self._detected_issues = [
            Issue(
                id=f"issue_{i}",
                type=IssueType.GEOMETRY_ERROR,
                description=f"Geometry issue {i}",
                severity=3,
                suggested_fix=f"Fix geometry issue {i}",
            )
            for i in range(1, count + 1)
        ]
        self._needs_refinement = True
        return self

    def with_refinement_iterations(self, iterations: int) -> "WorkflowStateBuilder":
        """Set refinement iteration count."""
        self._refinement_iterations = iterations
        return self

    def with_max_refinements_reached(self) -> "WorkflowStateBuilder":
        """Configure state as having reached max refinements."""
        self._refinement_iterations = self._max_refinements
        self._should_continue = False
        return self

    def with_user_acceptance(self, accepted: bool = True) -> "WorkflowStateBuilder":
        """Set user acceptance status."""
        self._user_accepted = accepted
        self._user_terminated = not accepted
        return self

    def with_error(self, error_message: str) -> "WorkflowStateBuilder":
        """Configure state with error condition."""
        self._error_message = error_message
        self._should_continue = False
        return self

    def build(self) -> WorkflowState:
        """Build the WorkflowState instance."""
        return WorkflowState(
            prompt=self._prompt,
            user_feedback=self._user_feedback,
            original_prompt=self._prompt,
            subtasks=self._subtasks,
            documentation=self._documentation,
            generated_code=self._generated_code,
            execution_result=self._execution_result,
            detected_issues=self._detected_issues,
            refinement_iterations=self._refinement_iterations,
            max_refinements=self._max_refinements,
            needs_refinement=self._needs_refinement,
            should_continue=self._should_continue,
            error_message=self._error_message,
            asset_metadata=self._asset_metadata,
            user_accepted=self._user_accepted,
            user_terminated=self._user_terminated,
        )


class CompositeWorkflowStateBuilder:
    """Builder for creating CompositeWorkflowState instances."""

    def __init__(self) -> None:
        """Initialize builder with default component states."""
        self._input_state = WorkflowInputState()
        self._processing_state = WorkflowProcessingState()
        self._analysis_state = WorkflowAnalysisState()
        self._refinement_state = WorkflowRefinementState()
        self._control_state = WorkflowControlState()
        self._asset_state = WorkflowAssetState()

    def with_input_state(self, input_state: WorkflowInputState) -> "CompositeWorkflowStateBuilder":
        """Set the input state component."""
        self._input_state = input_state
        return self

    def with_successful_processing(self) -> "CompositeWorkflowStateBuilder":
        """Configure processing state for successful execution."""
        self._processing_state = WorkflowProcessingState(
            generated_code="bpy.ops.mesh.primitive_cube_add()",
            execution_result=ExecutionResult(
                success=True,
                asset_path="/tmp/test_asset.blend",
                screenshot_path="/tmp/test_screenshot.png",
                logs=["Success"],
                errors=[],
                execution_time=2.0,
            ),
        )
        return self

    def with_analysis_issues(self, issue_count: int = 2) -> "CompositeWorkflowStateBuilder":
        """Configure analysis state with detected issues."""
        issues = [
            Issue(
                id=f"issue_{i}",
                type=IssueType.MATERIAL_ISSUE,
                description=f"Material problem {i}",
                severity=2,
                suggested_fix=f"Fix material issue {i}",
            )
            for i in range(issue_count)
        ]
        self._analysis_state = WorkflowAnalysisState(detected_issues=issues)
        return self

    def build(self) -> CompositeWorkflowState:
        """Build the CompositeWorkflowState instance."""
        return CompositeWorkflowState(
            input_state=self._input_state,
            processing_state=self._processing_state,
            analysis_state=self._analysis_state,
            refinement_state=self._refinement_state,
            control_state=self._control_state,
            asset_state=self._asset_state,
        )


class AgentResponseBuilder:
    """Builder for creating AgentResponse instances."""

    def __init__(self, agent_type: AgentType) -> None:
        """Initialize builder for specific agent type."""
        self._agent_type = agent_type
        self._success = True
        self._data: Any = None
        self._message = "Success"
        self._execution_time = 1.0
        self._metadata: dict[str, Any] = {}

    def with_success(self, data: Any = None) -> "AgentResponseBuilder":
        """Configure as successful response."""
        self._success = True
        self._data = data or self._get_default_success_data()
        self._message = "Operation completed successfully"
        return self

    def with_failure(self, error_message: str = "Operation failed") -> "AgentResponseBuilder":
        """Configure as failed response."""
        self._success = False
        self._data = self._get_default_failure_data()
        self._message = error_message
        return self

    def with_data(self, data: Any) -> "AgentResponseBuilder":
        """Set response data."""
        self._data = data
        return self

    def with_execution_time(self, seconds: float) -> "AgentResponseBuilder":
        """Set execution time."""
        self._execution_time = seconds
        return self

    def with_metadata(self, **metadata: Any) -> "AgentResponseBuilder":
        """Set response metadata."""
        self._metadata.update(metadata)
        return self

    def _get_default_success_data(self) -> Any:
        """Get default success data for agent type."""
        defaults = {
            AgentType.PLANNER: [SubTask(id="task1", type=TaskType.GEOMETRY, description="Create cube")],
            AgentType.RETRIEVAL: "# Blender documentation\\nbpy.ops.mesh.primitive_cube_add()",
            AgentType.CODING: "bpy.ops.mesh.primitive_cube_add()",
            AgentType.CRITIC: {"issues": [], "quality_score": 0.8},
            AgentType.VERIFICATION: {"valid": True, "metrics": {"polygon_count": 8}},
            AgentType.USER_PROXY: {"accepted": True, "feedback": None},
        }
        return defaults.get(self._agent_type, {})

    def _get_default_failure_data(self) -> Any:
        """Get default failure data for agent type."""
        defaults = {
            AgentType.PLANNER: [],
            AgentType.RETRIEVAL: "",
            AgentType.CODING: "",
            AgentType.CRITIC: {},
            AgentType.VERIFICATION: {},
            AgentType.USER_PROXY: {},
        }
        return defaults.get(self._agent_type, {})

    def build(self) -> AgentResponse:
        """Build the AgentResponse instance."""
        return AgentResponse(
            agent_type=self._agent_type,
            success=self._success,
            data=self._data,
            message=self._message,
            execution_time=self._execution_time,
            metadata=self._metadata,
        )


class BatchJobBuilder:
    """Builder for creating BatchJob instances."""

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._name = "Test Batch Job"
        self._user_id = uuid4()
        self._items: list[BatchJobItem] = []
        self._status = BatchStatus.QUEUED
        self._priority = BatchPriority.NORMAL
        self._created_at = datetime.utcnow()
        self._max_retries = 3
        self._parallel_workers = 3

    def with_name(self, name: str) -> "BatchJobBuilder":
        """Set job name."""
        self._name = name
        return self

    def with_user_id(self, user_id: str) -> "BatchJobBuilder":
        """Set user ID."""
        self._user_id = uuid4() if isinstance(user_id, str) else user_id
        return self

    def with_items(self, count: int = 3) -> "BatchJobBuilder":
        """Add items to the batch job."""
        self._items = [
            BatchJobItem(
                id=uuid4(),
                request=GenerateAssetRequest(
                    prompt=f"Create test asset {i}",
                    export_formats=["blend"],
                ),
            )
            for i in range(1, count + 1)
        ]
        return self

    def with_status(self, status: BatchStatus) -> "BatchJobBuilder":
        """Set job status."""
        self._status = status
        return self

    def with_priority(self, priority: BatchPriority) -> "BatchJobBuilder":
        """Set job priority."""
        self._priority = priority
        return self

    def with_completed_items(self, completed_count: int = 2) -> "BatchJobBuilder":
        """Configure some items as completed."""
        if not self._items:
            self.with_items(max(completed_count, 3))

        for i in range(min(completed_count, len(self._items))):
            self._items[i].status = BatchStatus.COMPLETED
            self._items[i].completed_at = datetime.utcnow()
            self._items[i].asset_id = uuid4()

        return self

    def with_failed_items(self, failed_count: int = 1) -> "BatchJobBuilder":
        """Configure some items as failed."""
        if not self._items:
            self.with_items(max(failed_count, 3))

        start_idx = len([item for item in self._items if item.status == BatchStatus.COMPLETED])

        for i in range(start_idx, min(start_idx + failed_count, len(self._items))):
            self._items[i].status = BatchStatus.FAILED
            self._items[i].completed_at = datetime.utcnow()
            self._items[i].error_message = f"Processing failed for item {i}"

        return self

    def build(self) -> BatchJob:
        """Build the BatchJob instance."""
        return BatchJob(
            id=uuid4(),
            name=self._name,
            user_id=self._user_id,
            items=self._items,
            status=self._status,
            priority=self._priority,
            created_at=self._created_at,
            max_retries=self._max_retries,
            parallel_workers=self._parallel_workers,
        )


class ExecutionResultBuilder:
    """Builder for creating ExecutionResult instances."""

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._success = True
        self._asset_path = None
        self._screenshot_path = None
        self._logs: list[str] = []
        self._errors: list[str] = []
        self._execution_time = 1.0

    def with_success(self, asset_path: str = "/tmp/test_asset.blend") -> "ExecutionResultBuilder":
        """Configure as successful execution."""
        self._success = True
        self._asset_path = asset_path
        self._screenshot_path = f"{asset_path}_screenshot.png"
        self._logs = ["Execution started", "Asset generated successfully"]
        self._errors = []
        return self

    def with_failure(self, error_message: str = "Execution failed") -> "ExecutionResultBuilder":
        """Configure as failed execution."""
        self._success = False
        self._asset_path = None
        self._screenshot_path = None
        self._logs = ["Execution started", "Error occurred"]
        self._errors = [error_message]
        return self

    def with_logs(self, *logs: str) -> "ExecutionResultBuilder":
        """Set execution logs."""
        self._logs = list(logs)
        return self

    def with_execution_time(self, seconds: float) -> "ExecutionResultBuilder":
        """Set execution time."""
        self._execution_time = seconds
        return self

    def build(self) -> ExecutionResult:
        """Build the ExecutionResult instance."""
        return ExecutionResult(
            success=self._success,
            asset_path=self._asset_path,
            screenshot_path=self._screenshot_path,
            logs=self._logs,
            errors=self._errors,
            execution_time=self._execution_time,
        )


# Convenience functions for quick creation
def create_successful_workflow_state(prompt: str = "Create a red cube") -> WorkflowState:
    """Create a WorkflowState with successful execution."""
    return WorkflowStateBuilder().with_prompt(prompt).with_successful_execution().build()


def create_failed_workflow_state(prompt: str = "Create invalid object", error: str = "Syntax error") -> WorkflowState:
    """Create a WorkflowState with failed execution."""
    return WorkflowStateBuilder().with_prompt(prompt).with_failed_execution(error).build()


def create_successful_agent_response(agent_type: AgentType, data: Any = None) -> AgentResponse:
    """Create a successful AgentResponse."""
    return AgentResponseBuilder(agent_type).with_success(data).build()


def create_failed_agent_response(agent_type: AgentType, error: str = "Agent failed") -> AgentResponse:
    """Create a failed AgentResponse."""
    return AgentResponseBuilder(agent_type).with_failure(error).build()
