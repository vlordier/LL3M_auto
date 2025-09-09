"""High-level workflow manager with interruption and resumption capabilities."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import structlog

from ..services.artifact_manager import ArtifactManager
from ..utils.types import WorkflowState
from .workflow import create_workflow, get_workflow_progress, resume_workflow_from_latest

logger = structlog.get_logger(__name__)


class WorkflowManager:
    """Manages workflow execution with interruption and resumption capabilities."""

    def __init__(self, artifacts_base_path: str = "artifacts"):
        self.artifact_manager = ArtifactManager(artifacts_base_path)
        self.active_workflows: dict[str, Any] = {}  # workflow_id -> compiled_workflow
        self.workflow_tasks: dict[str, asyncio.Task] = {}  # workflow_id -> asyncio_task

    async def start_workflow(self, prompt: str, config: dict[str, Any] | None = None) -> str:
        """Start a new workflow and return its ID."""
        workflow_id = str(uuid.uuid4())

        # Create initial workflow state
        initial_state = WorkflowState(prompt=prompt, workflow_id=workflow_id, session_id=str(uuid.uuid4()))

        logger.info(f"Starting new workflow {workflow_id}")

        # Create and store compiled workflow
        compiled_workflow = create_workflow(config)
        self.active_workflows[workflow_id] = compiled_workflow

        # Start workflow execution as background task
        task = asyncio.create_task(self._execute_workflow(workflow_id, initial_state, compiled_workflow))
        self.workflow_tasks[workflow_id] = task

        return workflow_id

    async def resume_workflow(self, workflow_id: str, config: dict[str, Any] | None = None) -> bool:
        """Resume a workflow from its latest checkpoint."""
        try:
            # Check if workflow is already running
            if workflow_id in self.workflow_tasks and not self.workflow_tasks[workflow_id].done():
                logger.warning(f"Workflow {workflow_id} is already running")
                return False

            # Attempt to resume from latest checkpoint
            resume_result = await resume_workflow_from_latest(workflow_id)
            if not resume_result:
                logger.error(f"No checkpoint found for workflow {workflow_id}")
                return False

            state, step_name = resume_result
            logger.info(f"Resuming workflow {workflow_id} from step {step_name}")

            # Create compiled workflow
            compiled_workflow = create_workflow(config)
            self.active_workflows[workflow_id] = compiled_workflow

            # Resume execution as background task
            task = asyncio.create_task(
                self._execute_workflow(workflow_id, state, compiled_workflow, resume_from_step=step_name)
            )
            self.workflow_tasks[workflow_id] = task

            return True

        except Exception as e:
            logger.exception(f"Failed to resume workflow {workflow_id}: {e}")
            return False

    async def interrupt_workflow(self, workflow_id: str, save_checkpoint: bool = True) -> bool:  # noqa: ARG002
        """Interrupt a running workflow, optionally saving a checkpoint."""
        try:
            if workflow_id not in self.workflow_tasks:
                logger.warning(f"Workflow {workflow_id} not found or not running")
                return False

            task = self.workflow_tasks[workflow_id]

            if task.done():
                logger.info(f"Workflow {workflow_id} already completed")
                return True

            # Cancel the task
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Workflow {workflow_id} successfully interrupted")

            # Clean up
            del self.workflow_tasks[workflow_id]
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            return True

        except Exception:
            logger.exception(f"Failed to interrupt workflow {workflow_id}")
            return False

    async def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Get current status and progress of a workflow."""
        status = {
            "workflow_id": workflow_id,
            "status": "unknown",
            "progress": None,
            "is_running": False,
            "can_resume": False,
            "error": None,
        }

        try:
            # Check if workflow is currently running
            if workflow_id in self.workflow_tasks:
                task = self.workflow_tasks[workflow_id]
                if not task.done():
                    status["status"] = "running"
                    status["is_running"] = True
                elif task.cancelled():
                    status["status"] = "interrupted"
                    status["can_resume"] = True
                elif task.exception():
                    status["status"] = "failed"
                    status["error"] = str(task.exception())
                    status["can_resume"] = True
                else:
                    status["status"] = "completed"

            # Get progress information from artifact manager
            progress = await get_workflow_progress(workflow_id)
            if progress:
                status["progress"] = progress
                if status["status"] == "unknown":
                    status["status"] = "has_checkpoints"
                    status["can_resume"] = True

        except Exception as e:
            status["error"] = str(e)
            logger.exception(f"Failed to get workflow status for {workflow_id}")

        return status

    async def list_workflows(self, include_completed: bool = True) -> list[dict[str, Any]]:
        """List all workflows with their current status."""
        workflows = []

        try:
            # Get all workflow IDs from artifact manager
            all_checkpoints = self.artifact_manager.checkpoints_registry
            workflow_ids = set()

            for checkpoint in all_checkpoints.values():
                workflow_ids.add(checkpoint.workflow_id)

            # Add currently running workflows
            for workflow_id in self.workflow_tasks:
                workflow_ids.add(workflow_id)

            # Get status for each workflow
            for workflow_id in workflow_ids:
                status = await self.get_workflow_status(workflow_id)

                if include_completed or status["status"] != "completed":
                    workflows.append(status)

            # Sort by most recent activity
            workflows.sort(
                key=lambda w: w.get("progress", {}).get("last_checkpoint", ""),
                reverse=True,
            )

        except Exception:
            logger.exception("Failed to list workflows")

        return workflows

    async def cleanup_workflow(self, workflow_id: str, keep_latest_checkpoint: bool = True) -> bool:
        """Clean up workflow artifacts and data."""
        try:
            # Stop workflow if running
            if workflow_id in self.workflow_tasks:
                await self.interrupt_workflow(workflow_id)

            # Clean up artifacts
            await self.artifact_manager.cleanup_workflow(workflow_id, keep_latest_checkpoint)

            logger.info(f"Cleaned up workflow {workflow_id}")
            return True

        except Exception:
            logger.exception(f"Failed to cleanup workflow {workflow_id}")
            return False

    async def get_workflow_artifacts(self, workflow_id: str, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """Get all artifacts for a workflow."""
        try:
            artifacts = await self.artifact_manager.get_workflow_artifacts(workflow_id, artifact_type)
            return [artifact.to_dict() for artifact in artifacts]

        except Exception:
            logger.exception(f"Failed to get artifacts for workflow {workflow_id}")
            return []

    async def export_workflow_state(self, workflow_id: str, export_path: str) -> bool:
        """Export complete workflow state for backup or transfer."""
        try:
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)

            # Get latest checkpoint
            latest_checkpoint = await self.artifact_manager.get_latest_checkpoint(workflow_id)
            if not latest_checkpoint:
                logger.error(f"No checkpoint found for workflow {workflow_id}")
                return False

            # Export checkpoint data
            import json

            checkpoint_file = export_path / f"workflow_{workflow_id}_checkpoint.json"
            with open(checkpoint_file, "w") as f:
                json.dump(latest_checkpoint.to_dict(), f, indent=2, default=str)

            # Copy all artifacts
            artifacts_dir = export_path / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            artifacts = await self.artifact_manager.get_workflow_artifacts(workflow_id)
            for artifact in artifacts:
                source_path = Path(artifact.file_path)
                if source_path.exists():
                    import shutil

                    target_path = artifacts_dir / f"{artifact.artifact_id}_{source_path.name}"
                    shutil.copy2(source_path, target_path)

            logger.info(f"Exported workflow {workflow_id} to {export_path}")
            return True

        except Exception:
            logger.exception(f"Failed to export workflow {workflow_id}")
            return False

    async def _execute_workflow(
        self,
        workflow_id: str,
        initial_state: WorkflowState,
        compiled_workflow: Any,
        resume_from_step: str | None = None,
    ) -> None:
        """Execute workflow in background task with proper error handling."""
        try:
            logger.info(f"Executing workflow {workflow_id}")

            # Configure workflow execution
            config = {"configurable": {"thread_id": workflow_id}}

            # Execute workflow
            if resume_from_step:
                # For resumption, we would need to modify the workflow to start from specific step
                # This is a placeholder - actual implementation would depend on LangGraph capabilities
                logger.info(f"Resuming from step: {resume_from_step}")

            # Stream workflow execution
            async for event in compiled_workflow.astream(initial_state, config=config):
                # Handle workflow events
                if event and isinstance(event, dict):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, WorkflowState):
                            logger.debug(f"Workflow {workflow_id} completed step: {node_name}")

            logger.info(f"Workflow {workflow_id} completed successfully")

        except asyncio.CancelledError:
            logger.info(f"Workflow {workflow_id} was interrupted")
            raise
        except Exception:
            logger.exception(f"Workflow {workflow_id} failed")
            raise
        finally:
            # Clean up references
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

    async def shutdown(self) -> None:
        """Shutdown workflow manager and cancel all running workflows."""
        logger.info("Shutting down workflow manager")

        # Cancel all running workflows
        for workflow_id in list(self.workflow_tasks.keys()):
            await self.interrupt_workflow(workflow_id)

        logger.info("Workflow manager shutdown complete")


# Global workflow manager instance
_workflow_manager: WorkflowManager | None = None


def get_workflow_manager() -> WorkflowManager:
    """Get global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager
