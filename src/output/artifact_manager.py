"""Enhanced artifact management for comprehensive output tracking and resumption."""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from ..utils.types import WorkflowArtifacts, WorkflowState

logger = structlog.get_logger(__name__)


class ArtifactManager:
    """Manages comprehensive artifact collection, organization, and resumption capabilities."""

    def __init__(self, base_output_dir: Path = Path("outputs")):
        """Initialize the artifact manager.

        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_output_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create structure
        self.sessions_dir = self.base_dir / "sessions"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.artifacts_dir = self.base_dir / "artifacts"

        for directory in [self.sessions_dir, self.checkpoints_dir, self.artifacts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def create_session(self, prompt: str, session_id: str | None = None) -> tuple[str, Path]:
        """Create a new generation session with comprehensive artifact tracking.

        Args:
            prompt: Original generation prompt
            session_id: Optional session ID (generated if None)

        Returns:
            Tuple of (session_id, session_directory)
        """
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive directory structure
        subdirs = {
            "artifacts": "All generated artifacts",
            "description": "Markdown documentation",
            "code": "Generated Python scripts",
            "assets": "3D asset files (.blend, .obj, etc.)",
            "renders": "Screenshots and rendered images",
            "exports": "Multi-format exports",
            "metadata": "Generation metadata and analysis",
            "refinements": "Refinement iteration history",
            "checkpoints": "Workflow resumption checkpoints",
        }

        for subdir, description in subdirs.items():
            (session_dir / subdir).mkdir(exist_ok=True)
            # Create README for each directory
            readme_file = session_dir / subdir / "README.md"
            readme_file.write_text(f"# {subdir.title()}\n\n{description}\n")

        # Create session metadata
        session_metadata = {
            "session_id": session_id,
            "prompt": prompt,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "directory_structure": subdirs,
            "total_artifacts": 0,
            "refinement_count": 0,
        }

        self._save_json(session_dir / "metadata" / "session.json", session_metadata)

        logger.info(f"Created session: {session_id}", session_dir=str(session_dir))
        return session_id, session_dir

    def save_comprehensive_artifacts(
        self,
        session_dir: Path,
        workflow_state: WorkflowState,
        generated_code: str | None = None,
        asset_files: list[str] | None = None,
        render_files: list[str] | None = None,
        export_formats: dict[str, str] | None = None,
    ) -> WorkflowArtifacts:
        """Save comprehensive artifacts with full documentation.

        Args:
            session_dir: Session directory
            workflow_state: Current workflow state
            generated_code: Generated Blender Python code
            asset_files: List of generated asset file paths
            render_files: List of render/screenshot file paths
            export_formats: Dictionary of format -> file path mappings

        Returns:
            Complete WorkflowArtifacts object
        """
        artifacts = WorkflowArtifacts()

        # 1. Save detailed description as markdown
        description_file = self._create_description_file(session_dir, workflow_state)
        artifacts.description_file = str(description_file)

        # 2. Save and organize generated code
        if generated_code:
            code_file = session_dir / "code" / "generated_script.py"
            code_file.write_text(generated_code)
            artifacts.generated_code = str(code_file)

            # Also save with metadata
            code_metadata = {
                "generated_at": datetime.now().isoformat(),
                "prompt": workflow_state.prompt,
                "refinement_iteration": workflow_state.refinement_iterations,
                "code_length": len(generated_code),
            }
            self._save_json(session_dir / "code" / "code_metadata.json", code_metadata)

        # 3. Organize asset files
        if asset_files:
            for asset_file in asset_files:
                if Path(asset_file).exists():
                    dest_path = session_dir / "assets" / Path(asset_file).name
                    shutil.copy2(asset_file, dest_path)
                    artifacts.asset_files.append(str(dest_path))

        # 4. Organize render files
        if render_files:
            for render_file in render_files:
                if Path(render_file).exists():
                    dest_path = session_dir / "renders" / Path(render_file).name
                    shutil.copy2(render_file, dest_path)
                    artifacts.render_files.append(str(dest_path))

        # 5. Organize export files
        if export_formats:
            for format_name, file_path in export_formats.items():
                if Path(file_path).exists():
                    dest_path = session_dir / "exports" / f"{workflow_state.session_id}.{format_name}"
                    shutil.copy2(file_path, dest_path)
                    artifacts.export_files[format_name] = str(dest_path)

        # 6. Save comprehensive generation metadata
        generation_metadata = {
            "session_id": workflow_state.session_id,
            "prompt": workflow_state.prompt,
            "original_prompt": workflow_state.original_prompt,
            "refinement_iterations": workflow_state.refinement_iterations,
            "generation_timestamp": datetime.now().isoformat(),
            "subtasks": [task.model_dump() for task in workflow_state.subtasks],
            "execution_result": (
                workflow_state.execution_result.model_dump() if workflow_state.execution_result else None
            ),
            "detected_issues": [issue.model_dump() for issue in workflow_state.detected_issues],
        }
        artifacts.generation_metadata = generation_metadata
        self._save_json(session_dir / "metadata" / "generation.json", generation_metadata)

        # 7. Save quality analysis
        if workflow_state.critic_analysis:
            artifacts.quality_analysis = workflow_state.critic_analysis
            self._save_json(session_dir / "metadata" / "quality_analysis.json", workflow_state.critic_analysis)

        # 8. Save refinement history
        if workflow_state.refinement_iterations > 0:
            refinement_entry = {
                "iteration": workflow_state.refinement_iterations,
                "timestamp": datetime.now().isoformat(),
                "user_feedback": workflow_state.user_feedback,
                "refinement_request": workflow_state.refinement_request,
                "quality_before": (
                    workflow_state.critic_analysis.get("overall_score")
                    if workflow_state.critic_analysis else None
                ),
            }
            artifacts.refinement_history.append(refinement_entry)

            refinement_file = (
                session_dir / "refinements" / f"refinement_{workflow_state.refinement_iterations:02d}.json"
            )
            self._save_json(refinement_file, refinement_entry)

        # 9. Generate resume prompt
        artifacts.resume_prompt = workflow_state.generate_resume_prompt()

        # 10. Save workflow checkpoint for resumption
        checkpoint_data = workflow_state.save_artifacts_checkpoint(session_dir / "checkpoints")
        artifacts.workflow_checkpoint = checkpoint_data

        # Update workflow state with artifacts
        workflow_state.artifacts = artifacts

        logger.info("Saved comprehensive artifacts",
                   session_id=workflow_state.session_id,
                   artifact_count=len(
                       artifacts.asset_files + artifacts.render_files + list(artifacts.export_files.values())
                   ))

        return artifacts

    def load_session_for_resumption(self, session_id: str) -> tuple[WorkflowState, Path]:
        """Load a session for resumption from checkpoint.

        Args:
            session_id: Session identifier to resume

        Returns:
            Tuple of (restored_workflow_state, session_directory)

        Raises:
            FileNotFoundError: If session or checkpoint not found
        """
        session_dir = self.sessions_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        checkpoint_file = session_dir / "checkpoints" / "workflow_checkpoint.json"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint found for session: {session_id}")

        # Load checkpoint data
        checkpoint_data = self._load_json(checkpoint_file)

        # Reconstruct WorkflowState
        workflow_state = WorkflowState(
            session_id=checkpoint_data["session_id"],
            prompt=checkpoint_data["prompt"],
            original_prompt=checkpoint_data["original_prompt"],
            refinement_iterations=checkpoint_data["refinement_iterations"],
            user_feedback=checkpoint_data["workflow_state"]["user_feedback"],
            critic_analysis=checkpoint_data["workflow_state"]["critic_analysis"],
            verification_result=checkpoint_data["workflow_state"]["verification_result"],
            should_continue=checkpoint_data["workflow_state"]["should_continue"],
            output_directory=str(session_dir),
        )

        # Restore artifacts
        artifacts_data = checkpoint_data["artifacts"]
        workflow_state.artifacts = WorkflowArtifacts(**artifacts_data)

        logger.info(f"Loaded session for resumption: {session_id}",
                   checkpoint_timestamp=checkpoint_data["timestamp"])

        return workflow_state, session_dir

    def create_resumption_prompt(self, session_id: str) -> str:
        """Create a human-readable resumption prompt.

        Args:
            session_id: Session to create resumption prompt for

        Returns:
            Formatted resumption prompt
        """
        try:
            workflow_state, session_dir = self.load_session_for_resumption(session_id)

            resume_prompt = f"""
# Resume 3D Generation Session

**Session ID:** {session_id}
**Original Prompt:** {workflow_state.original_prompt or workflow_state.prompt}
**Refinement Iterations:** {workflow_state.refinement_iterations}

## Current Status
{workflow_state.artifacts.get_complete_artifact_summary()}

## Quality Assessment
"""
            if workflow_state.critic_analysis:
                overall_score = workflow_state.critic_analysis.get('overall_score', 'N/A')
                resume_prompt += f"Overall Quality Score: {overall_score}/10\n"

                if 'critical_issues' in workflow_state.critic_analysis:
                    issues = workflow_state.critic_analysis['critical_issues']
                    if issues:
                        resume_prompt += f"Outstanding Issues: {', '.join(issues)}\n"

            if workflow_state.user_feedback:
                resume_prompt += f"\n## Last User Feedback\n{workflow_state.user_feedback}\n"

            resume_prompt += f"\n## To Resume\nUse the session ID `{session_id}` to continue this generation session."

            return resume_prompt.strip()

        except Exception:
            logger.exception("Failed to create resumption prompt")
            return f"Error: Could not load session {session_id} for resumption."

    def list_resumable_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """List sessions that can be resumed.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session information dictionaries
        """
        resumable = []

        for session_dir in sorted(self.sessions_dir.iterdir(),
                                 key=lambda p: p.stat().st_mtime, reverse=True):
            if not session_dir.is_dir():
                continue

            checkpoint_file = session_dir / "checkpoints" / "workflow_checkpoint.json"
            if not checkpoint_file.exists():
                continue

            try:
                checkpoint_data = self._load_json(checkpoint_file)
                session_info = {
                    "session_id": session_dir.name,
                    "prompt": checkpoint_data.get("prompt", "Unknown"),
                    "refinement_iterations": checkpoint_data.get("refinement_iterations", 0),
                    "last_updated": checkpoint_data.get("timestamp", "Unknown"),
                    "artifacts_count": len(
                        checkpoint_data.get("artifacts", {}).get("asset_files", []) +
                        checkpoint_data.get("artifacts", {}).get("render_files", [])
                    ),
                    "can_resume": checkpoint_data.get("workflow_state", {}).get("should_continue", False),
                }
                resumable.append(session_info)

                if len(resumable) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Could not load session {session_dir.name}: {e}")
                continue

        return resumable

    def _create_description_file(self, session_dir: Path, workflow_state: WorkflowState) -> Path:
        """Create a comprehensive markdown description file."""
        description_file = session_dir / "description" / "README.md"

        content = f"""# 3D Asset Generation Session

## Overview
**Session ID:** {workflow_state.session_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** {'Completed' if not workflow_state.should_continue else 'In Progress'}

## Original Prompt
```
{workflow_state.prompt}
```

## Generation Details
- **Refinement Iterations:** {workflow_state.refinement_iterations}
- **Subtasks Completed:** {len(workflow_state.subtasks)}
- **Issues Detected:** {len(workflow_state.detected_issues)}
"""

        # Add quality assessment
        if workflow_state.critic_analysis:
            overall_score = workflow_state.critic_analysis.get('overall_score', 'N/A')
            content += f"\n## Quality Assessment\n- **Overall Score:** {overall_score}/10\n"

            if 'visual_quality' in workflow_state.critic_analysis:
                for dimension, data in workflow_state.critic_analysis.items():
                    if isinstance(data, dict) and 'score' in data:
                        content += f"- **{dimension.replace('_', ' ').title()}:** {data['score']}/10\n"

        # Add subtasks
        if workflow_state.subtasks:
            content += "\n## Completed Subtasks\n"
            for i, task in enumerate(workflow_state.subtasks, 1):
                content += f"{i}. **{task.type.value.title()}:** {task.description}\n"

        # Add issues if any
        if workflow_state.detected_issues:
            content += "\n## Detected Issues\n"
            for issue in workflow_state.detected_issues:
                content += f"- **{issue.type.value.title()}** (Severity {issue.severity}): {issue.description}\n"

        # Add user feedback history
        if workflow_state.user_feedback:
            content += f"\n## Latest User Feedback\n```\n{workflow_state.user_feedback}\n```\n"

        # Add file structure
        content += "\n## Generated Files\n"
        content += "```\n"
        content += f"{session_dir.name}/\n"
        content += "├── code/           # Generated Python scripts\n"
        content += "├── assets/         # 3D asset files\n"
        content += "├── renders/        # Screenshots and renders\n"
        content += "├── exports/        # Multi-format exports\n"
        content += "├── metadata/       # Generation metadata\n"
        content += "├── refinements/    # Refinement history\n"
        content += "└── checkpoints/    # Resumption checkpoints\n"
        content += "```\n"

        # Add resumption info
        content += f"\n## Resumption\nTo resume this session, use session ID: `{workflow_state.session_id}`\n"

        description_file.write_text(content)
        return description_file

    def _save_json(self, file_path: Path, data: dict[str, Any]) -> None:
        """Save data to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path) as f:
            return json.load(f)
