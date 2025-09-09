"""Session Integration Hooks for Workflow Tracing

Integrates the stateful session manager with existing workflow components
to provide comprehensive traceability and resumption capabilities.
"""

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from .stateful_session_manager import (
    ArtifactType,
    CodeVersion,
    ExecutionTrace,
    SessionArtifact,
    SessionPhase,
    SessionState,
    StatefulSessionManager,
)

logger = structlog.get_logger(__name__)


class SessionTracker:
    """Tracks workflow execution within a stateful session."""

    def __init__(self, session_manager: StatefulSessionManager, session_state: SessionState):  # noqa: D107
        self.session_manager = session_manager
        self.session_state = session_state
        self.current_trace: ExecutionTrace | None = None

    def start_phase(self, phase: SessionPhase, inputs: dict[str, Any] = None) -> ExecutionTrace:
        """Start tracking a new execution phase."""
        # End current trace if exists
        if self.current_trace and not self.current_trace.end_time:
            self.end_current_trace(status="interrupted")

        # Create new trace
        trace = ExecutionTrace(
            phase=phase,
            start_time=datetime.now(UTC),
            end_time=None,
            duration_seconds=None,
            status="running",
            inputs=inputs or {},
            outputs={},
            errors=[],
            logs=[],
            llm_calls=[],
            agent_actions=[]
        )

        self.current_trace = trace
        self.session_state.traces.append(trace)
        self.session_state.current_phase = phase

        # Update completion percentage based on phase
        completion_map = {
            SessionPhase.INITIALIZATION: 5,
            SessionPhase.PLANNING: 15,
            SessionPhase.RETRIEVAL: 25,
            SessionPhase.CODE_GENERATION: 50,
            SessionPhase.CODE_VALIDATION: 60,
            SessionPhase.CODE_HEALING: 70,
            SessionPhase.EXECUTION: 85,
            SessionPhase.QUALITY_ASSESSMENT: 95,
            SessionPhase.COMPLETION: 100
        }
        self.session_state.completion_percentage = completion_map.get(phase, 0)

        logger.info(f"Started phase {phase.value}",
                   session_id=self.session_state.session_id)

        return trace

    def end_current_trace(self, status: str = "completed", outputs: dict[str, Any] = None):
        """End the current execution trace."""
        if not self.current_trace:
            return

        now = datetime.now(UTC)
        self.current_trace.end_time = now
        self.current_trace.status = status
        self.current_trace.duration_seconds = (now - self.current_trace.start_time).total_seconds()

        if outputs:
            self.current_trace.outputs.update(outputs)

        logger.debug(f"Ended trace for {self.current_trace.phase.value}",
                    duration=self.current_trace.duration_seconds,
                    status=status)

        self.current_trace = None
        self._save_session()

    def log_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                     execution_time: float, success: bool, error: str = None):
        """Log an LLM API call."""
        if not self.current_trace:
            return

        llm_call = {
            'timestamp': datetime.now(UTC).isoformat(),
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'execution_time': execution_time,
            'success': success,
            'error': error
        }

        self.current_trace.llm_calls.append(llm_call)

    def log_agent_action(self, agent_type: str, action: str, inputs: dict[str, Any],
                        outputs: dict[str, Any], success: bool, error: str = None):
        """Log an agent action."""
        if not self.current_trace:
            return

        agent_action = {
            'timestamp': datetime.now(UTC).isoformat(),
            'agent_type': agent_type,
            'action': action,
            'inputs': inputs,
            'outputs': outputs,
            'success': success,
            'error': error
        }

        self.current_trace.agent_actions.append(agent_action)

    def add_code_version(self, code_content: str, source_phase: SessionPhase,
                        validation_status: str, ruff_issues: list[str] = None,
                        execution_errors: list[str] = None, healing_applied: bool = False,
                        quality_score: float = None):
        """Add a new code version to the session."""
        version = len(self.session_state.code_versions) + 1

        code_version = CodeVersion(
            version=version,
            timestamp=datetime.now(UTC),
            code_content=code_content,
            source_phase=source_phase,
            validation_status=validation_status,
            ruff_issues=ruff_issues or [],
            execution_errors=execution_errors or [],
            healing_applied=healing_applied,
            quality_score=quality_score
        )

        self.session_state.code_versions.append(code_version)

        # Save code version to file
        session_dir = self.session_manager.get_session_directory(self.session_state.session_id)
        code_file = session_dir / "artifacts" / "code_versions" / f"version_{version:03d}.py"

        with open(code_file, 'w') as f:
            f.write(code_content)

        # Create artifact record
        self.store_artifact(
            artifact_type=ArtifactType.CODE,
            file_path=str(code_file),
            phase=source_phase,
            version=version,
            metadata={
                'validation_status': validation_status,
                'ruff_issues_count': len(ruff_issues or []),
                'execution_errors_count': len(execution_errors or []),
                'healing_applied': healing_applied,
                'quality_score': quality_score
            }
        )

        logger.info(f"Added code version {version}",
                   validation_status=validation_status,
                   session_id=self.session_state.session_id)

    def store_artifact(self, artifact_type: ArtifactType, file_path: str,
                      phase: SessionPhase, version: int = 1,
                      dependencies: list[str] = None, tags: list[str] = None,
                      metadata: dict[str, Any] = None) -> str:
        """Store an artifact with comprehensive metadata."""
        artifact_id = str(uuid.uuid4())
        file_path_obj = Path(file_path)

        # Calculate checksum
        checksum = ""
        size_bytes = 0
        if file_path_obj.exists():
            with open(file_path_obj, 'rb') as f:
                content = f.read()
                checksum = hashlib.sha256(content).hexdigest()
                size_bytes = len(content)

        # Create relative path from session directory
        session_dir = self.session_manager.get_session_directory(self.session_state.session_id)
        try:
            relative_path = str(file_path_obj.relative_to(session_dir))
        except ValueError:
            # File is outside session directory
            relative_path = str(file_path_obj)

        artifact = SessionArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            file_path=str(file_path_obj),
            relative_path=relative_path,
            created_at=datetime.now(UTC),
            phase=phase,
            version=version,
            checksum=checksum,
            size_bytes=size_bytes,
            dependencies=dependencies or [],
            tags=tags or [],
            metadata=metadata or {}
        )

        self.session_state.artifacts.append(artifact)

        logger.info(f"Stored artifact {artifact_id}",
                   type=artifact_type.value,
                   phase=phase.value,
                   size_bytes=size_bytes)

        return artifact_id

    def add_log(self, message: str, level: str = "info"):
        """Add a log message to the current trace."""
        if not self.current_trace:
            return

        log_entry = f"[{datetime.now(UTC).isoformat()}] [{level.upper()}] {message}"
        self.current_trace.logs.append(log_entry)

    def add_error(self, error: str):
        """Add an error to the current trace."""
        if not self.current_trace:
            return

        self.current_trace.errors.append(error)
        self.session_state.current_phase = SessionPhase.ERROR

    def _save_session(self):
        """Save the current session state."""
        self.session_manager.save_session(self.session_state)


def create_session_tracker(prompt: str, asset_name: str, output_directory: str) -> SessionTracker:
    """Create a new session tracker for a workflow."""
    session_manager = StatefulSessionManager()
    session_state = session_manager.create_session(prompt, asset_name, output_directory)
    return SessionTracker(session_manager, session_state)


def resume_session_tracker(session_id: str) -> SessionTracker | None:
    """Resume an existing session tracker."""
    session_manager = StatefulSessionManager()
    session_state = session_manager.load_session(session_id)

    if not session_state:
        return None

    return SessionTracker(session_manager, session_state)
