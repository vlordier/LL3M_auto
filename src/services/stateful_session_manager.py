"""Comprehensive Stateful Session Management System

This system provides:
- Complete execution traceability
- Version-controlled artifact management
- Resumable workflow sessions
- Comprehensive inspection capabilities
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SessionPhase(Enum):
    """Workflow execution phases for detailed tracking."""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    CODE_HEALING = "code_healing"
    EXECUTION = "execution"
    QUALITY_ASSESSMENT = "quality_assessment"
    REFINEMENT = "refinement"
    COMPLETION = "completion"
    ERROR = "error"
    PAUSED = "paused"
    STEPPING = "stepping"


class ExecutionMode(Enum):
    """Execution modes for step control."""
    CONTINUOUS = "continuous"  # Run all phases automatically
    STEP_BY_STEP = "step_by_step"  # Pause after each phase
    PAUSED = "paused"  # Currently paused
    STEPPING = "stepping"  # Execute single step then pause


@dataclass
class SubstepProgress:
    """Granular substep within a phase."""
    substep_id: str
    name: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed', 'skipped'
    progress_percent: float
    start_time: datetime | None = None
    end_time: datetime | None = None
    technical_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class AgentReflection:
    """Agent's reflection on its decision with confidence scoring."""
    reflection_id: str
    agent_name: str
    phase: SessionPhase
    timestamp: datetime
    reflection_text: str
    confidence_score: float  # 0.0 to 1.0
    decision_made: str
    reasoning: str
    alternative_approaches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'phase': self.phase.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class InteractionPoint:
    """Interactive point where user input is needed."""
    interaction_id: str
    phase: SessionPhase
    question: str
    interaction_type: str
    created_at: datetime
    options: list[str] = field(default_factory=list)
    timeout_seconds: int | None = None
    is_resolved: bool = False
    response: str | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'phase': self.phase.value,
            'created_at': self.created_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class ProgressMessage:
    """Detailed progress message for user-friendly updates."""
    timestamp: datetime
    phase: SessionPhase
    title: str
    description: str
    progress_percent: float  # 0-100
    estimated_remaining_seconds: float | None = None
    technical_details: dict[str, Any] | None = None

    # Enhanced granular tracking
    substeps: list[SubstepProgress] | None = None
    interaction_points: list[InteractionPoint] | None = None
    agent_reflections: list[AgentReflection] | None = None
    preview_available: bool = False
    preview_url: str | None = None

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'phase': self.phase.value,
            'substeps': [s.to_dict() for s in (self.substeps or [])] if self.substeps else [],
            'interaction_points': [i.to_dict() for i in (self.interaction_points or [])] if self.interaction_points else [],  # noqa: E501
            'agent_reflections': [a.to_dict() for a in (self.agent_reflections or [])] if self.agent_reflections else []
        }


@dataclass
class StepControl:
    """Controls for step-by-step execution."""
    execution_mode: ExecutionMode
    current_step: int
    total_steps: int
    can_pause: bool
    can_resume: bool
    can_step: bool
    awaiting_user_action: bool

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'execution_mode': self.execution_mode.value
        }


class ArtifactType(Enum):
    """Types of artifacts generated during workflow."""
    BLEND_FILE = "blend_file"
    SCREENSHOT = "screenshot"
    CODE = "code"
    LOGS = "logs"
    TRACE = "trace"
    STATE = "state"
    CHECKPOINT = "checkpoint"
    EXPORTS = "exports"
    METADATA = "metadata"


@dataclass
class CodeVersion:
    """Tracks evolution of generated code."""
    version: int
    timestamp: datetime
    code_content: str
    source_phase: SessionPhase
    validation_status: str  # 'valid', 'invalid', 'healing', 'final'
    healing_applied: bool
    ruff_issues: list[str] = field(default_factory=list)
    execution_errors: list[str] = field(default_factory=list)
    quality_score: float | None = None

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'source_phase': self.source_phase.value
        }


@dataclass
class ExecutionTrace:
    """Detailed execution trace for each phase."""
    phase: SessionPhase
    start_time: datetime
    end_time: datetime | None
    duration_seconds: float | None
    status: str  # 'running', 'completed', 'failed', 'skipped'
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)  # Track all LLM interactions
    agent_actions: list[dict[str, Any]] = field(default_factory=list)  # Track agent-specific actions

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'phase': self.phase.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class SessionArtifact:
    """Enhanced artifact with comprehensive metadata."""
    artifact_id: str
    artifact_type: ArtifactType
    file_path: str
    relative_path: str
    created_at: datetime
    phase: SessionPhase
    version: int
    checksum: str
    size_bytes: int
    metadata: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)  # IDs of dependent artifacts
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            **asdict(self),
            'artifact_type': self.artifact_type.value,
            'created_at': self.created_at.isoformat(),
            'phase': self.phase.value
        }


@dataclass
class SessionState:
    """Complete session state for resumption."""
    session_id: str
    prompt: str
    asset_name: str
    output_directory: str
    created_at: datetime
    updated_at: datetime
    current_phase: SessionPhase
    completion_percentage: float
    step_control: StepControl
    success: bool

    # Execution history
    traces: list[ExecutionTrace] = field(default_factory=list)
    code_versions: list[CodeVersion] = field(default_factory=list)
    artifacts: list[SessionArtifact] = field(default_factory=list)

    # Progress tracking
    progress_messages: list[ProgressMessage] = field(default_factory=list)

    # Workflow state
    workflow_state: dict[str, Any] | None = None
    checkpoints: list[str] = field(default_factory=list)  # Checkpoint IDs for resumption

    # Results
    final_result: dict[str, Any] | None = None
    total_duration: float | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            'session_id': self.session_id,
            'prompt': self.prompt,
            'asset_name': self.asset_name,
            'output_directory': self.output_directory,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'current_phase': self.current_phase.value,
            'completion_percentage': self.completion_percentage,
            'step_control': self.step_control.to_dict(),
            'success': self.success,
            'traces': [trace.to_dict() for trace in self.traces],
            'code_versions': [version.to_dict() for version in self.code_versions],
            'artifacts': [artifact.to_dict() for artifact in self.artifacts],
            'progress_messages': [msg.to_dict() for msg in self.progress_messages],
            'workflow_state': self.workflow_state,
            'checkpoints': self.checkpoints,
            'final_result': self.final_result,
            'total_duration': self.total_duration,
            'tags': self.tags,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SessionState':
        """Reconstruct SessionState from dictionary."""
        # Convert string enums back to enum objects
        traces = [
            ExecutionTrace(
                phase=SessionPhase(trace_data['phase']),
                start_time=datetime.fromisoformat(trace_data['start_time']),
                end_time=datetime.fromisoformat(trace_data['end_time']) if trace_data['end_time'] else None,
                duration_seconds=trace_data['duration_seconds'],
                status=trace_data['status'],
                inputs=trace_data['inputs'],
                outputs=trace_data['outputs'],
                errors=trace_data['errors'],
                logs=trace_data['logs'],
                llm_calls=trace_data['llm_calls'],
                agent_actions=trace_data['agent_actions']
            )
            for trace_data in data['traces']
        ]

        code_versions = [
            CodeVersion(
                version=version_data['version'],
                timestamp=datetime.fromisoformat(version_data['timestamp']),
                code_content=version_data['code_content'],
                source_phase=SessionPhase(version_data['source_phase']),
                validation_status=version_data['validation_status'],
                ruff_issues=version_data['ruff_issues'],
                execution_errors=version_data['execution_errors'],
                healing_applied=version_data['healing_applied'],
                quality_score=version_data.get('quality_score')
            )
            for version_data in data['code_versions']
        ]

        artifacts = [
            SessionArtifact(
                artifact_id=artifact_data['artifact_id'],
                artifact_type=ArtifactType(artifact_data['artifact_type']),
                file_path=artifact_data['file_path'],
                relative_path=artifact_data['relative_path'],
                created_at=datetime.fromisoformat(artifact_data['created_at']),
                phase=SessionPhase(artifact_data['phase']),
                version=artifact_data['version'],
                checksum=artifact_data['checksum'],
                size_bytes=artifact_data['size_bytes'],
                dependencies=artifact_data['dependencies'],
                tags=artifact_data['tags'],
                metadata=artifact_data['metadata']
            )
            for artifact_data in data['artifacts']
        ]

        # Reconstruct step control (use default if missing for backward compatibility)
        step_control_data = data.get('step_control', {})
        step_control = StepControl(
            execution_mode=ExecutionMode(step_control_data.get('execution_mode', 'continuous')),
            current_step=step_control_data.get('current_step', 0),
            total_steps=step_control_data.get('total_steps', 10),
            can_pause=step_control_data.get('can_pause', True),
            can_resume=step_control_data.get('can_resume', False),
            can_step=step_control_data.get('can_step', True),
            awaiting_user_action=step_control_data.get('awaiting_user_action', False)
        )

        # Reconstruct progress messages (use empty list if missing)
        progress_messages = []
        if 'progress_messages' in data:
            for msg_data in data['progress_messages']:
                progress_messages.append(ProgressMessage(
                    timestamp=datetime.fromisoformat(msg_data['timestamp']),
                    phase=SessionPhase(msg_data['phase']),
                    title=msg_data['title'],
                    description=msg_data['description'],
                    progress_percent=msg_data['progress_percent'],
                    estimated_remaining_seconds=msg_data.get('estimated_remaining_seconds'),
                    technical_details=msg_data.get('technical_details')
                ))

        return cls(
            session_id=data['session_id'],
            prompt=data['prompt'],
            asset_name=data['asset_name'],
            output_directory=data['output_directory'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            current_phase=SessionPhase(data['current_phase']),
            completion_percentage=data['completion_percentage'],
            step_control=step_control,
            success=data['success'],
            traces=traces,
            code_versions=code_versions,
            artifacts=artifacts,
            progress_messages=progress_messages,
            workflow_state=data.get('workflow_state'),
            checkpoints=data.get('checkpoints', []),
            final_result=data.get('final_result'),
            total_duration=data.get('total_duration'),
            tags=data.get('tags', []),
            notes=data.get('notes', '')
        )


class StatefulSessionManager:
    """Manages stateful sessions with comprehensive traceability."""

    def __init__(self, base_output_dir: str = "outputs"):  # noqa: D107
        self.base_output_dir = Path(base_output_dir)
        self.sessions_dir = self.base_output_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Session registry
        self.sessions_registry_path = self.sessions_dir / "sessions_registry.json"
        self.sessions_registry = self._load_sessions_registry()

    def _load_sessions_registry(self) -> dict[str, dict[str, Any]]:
        """Load sessions registry from disk."""
        if self.sessions_registry_path.exists():
            with open(self.sessions_registry_path) as f:
                return json.load(f)
        return {}

    def _save_sessions_registry(self):
        """Save sessions registry to disk."""
        with open(self.sessions_registry_path, 'w') as f:
            json.dump(self.sessions_registry, f, indent=2)

    def create_session(self, prompt: str, asset_name: str, output_directory: str) -> SessionState:
        """Create a new stateful session."""
        session_id = f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        now = datetime.now(UTC)

        # Create session directory structure
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create organized subdirectories
        (session_dir / "artifacts").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "code_versions").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "screenshots").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "blends").mkdir(exist_ok=True)
        (session_dir / "artifacts" / "exports").mkdir(exist_ok=True)
        (session_dir / "traces").mkdir(exist_ok=True)
        (session_dir / "checkpoints").mkdir(exist_ok=True)
        (session_dir / "logs").mkdir(exist_ok=True)

        # Initialize session state with progress tracking
        initial_step_control = StepControl(
            execution_mode=ExecutionMode.CONTINUOUS,
            current_step=0,
            total_steps=10,  # Standard workflow phases
            can_pause=True,
            can_resume=False,
            can_step=True,
            awaiting_user_action=False
        )

        initial_progress_message = ProgressMessage(
            timestamp=now,
            phase=SessionPhase.INITIALIZATION,
            title="🚀 Starting 3D Asset Generation",
            description="Setting up your asset generation workflow...",
            progress_percent=0.0,
            estimated_remaining_seconds=180.0,  # 3 minutes estimate
            technical_details={"prompt": prompt, "asset_name": asset_name}
        )

        session_state = SessionState(
            session_id=session_id,
            prompt=prompt,
            asset_name=asset_name,
            output_directory=output_directory,
            created_at=now,
            updated_at=now,
            current_phase=SessionPhase.INITIALIZATION,
            completion_percentage=0.0,
            traces=[],
            code_versions=[],
            artifacts=[],
            progress_messages=[initial_progress_message],
            step_control=initial_step_control,
            workflow_state=None,
            checkpoints=[],
            final_result=None,
            success=False,
            total_duration=None,
            tags=[],
            notes=""
        )

        # Save initial state
        self.save_session(session_state)

        # Update registry
        self.sessions_registry[session_id] = {
            'prompt': prompt,
            'asset_name': asset_name,
            'created_at': now.isoformat(),
            'current_phase': SessionPhase.INITIALIZATION.value,
            'success': False
        }
        self._save_sessions_registry()

        logger.info(f"Created session {session_id}",
                   prompt=prompt, asset_name=asset_name)

        return session_state

    def save_session(self, session_state: SessionState):
        """Save session state to disk."""
        session_dir = self.sessions_dir / session_state.session_id
        session_file = session_dir / "session_state.json"

        session_state.updated_at = datetime.now(UTC)

        with open(session_file, 'w') as f:
            json.dump(session_state.to_dict(), f, indent=2)

        # Update registry (ensure entry exists first)
        if session_state.session_id not in self.sessions_registry:
            self.sessions_registry[session_state.session_id] = {
                'prompt': session_state.prompt,
                'asset_name': session_state.asset_name,
                'created_at': session_state.created_at.isoformat(),
            }

        self.sessions_registry[session_state.session_id].update({
            'current_phase': session_state.current_phase.value,
            'completion_percentage': session_state.completion_percentage,
            'success': session_state.success,
            'updated_at': session_state.updated_at.isoformat()
        })
        self._save_sessions_registry()

        logger.debug(f"Saved session {session_state.session_id}")

    def load_session(self, session_id: str) -> SessionState | None:
        """Load session state from disk."""
        session_file = self.sessions_dir / session_id / "session_state.json"

        if not session_file.exists():
            return None

        with open(session_file) as f:
            data = json.load(f)

        return SessionState.from_dict(data)

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all sessions with basic metadata."""
        sessions = []
        for session_id, metadata in self.sessions_registry.items():
            sessions.append({
                'session_id': session_id,
                **metadata
            })

        # Sort by creation time, most recent first
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        return sessions[:limit]

    def get_session_directory(self, session_id: str) -> Path:
        """Get the directory path for a session."""
        return self.sessions_dir / session_id

    # Progress Tracking and Step Control Methods

    def add_progress_message(self, session_id: str, phase: SessionPhase, title: str,
                           description: str, progress_percent: float,
                           estimated_remaining_seconds: float | None = None,
                           technical_details: dict[str, Any] | None = None):
        """Add a detailed progress message to the session."""
        session = self.load_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for progress update")
            return

        progress_message = ProgressMessage(
            timestamp=datetime.now(UTC),
            phase=phase,
            title=title,
            description=description,
            progress_percent=progress_percent,
            estimated_remaining_seconds=estimated_remaining_seconds,
            technical_details=technical_details or {}
        )

        session.progress_messages.append(progress_message)
        session.completion_percentage = progress_percent
        session.current_phase = phase

        self.save_session(session)

        logger.info(f"📊 Progress Update: {title}",
                   session_id=session_id,
                   phase=phase.value,
                   progress=f"{progress_percent:.1f}%")

    def update_step_control(self, session_id: str, execution_mode: ExecutionMode,
                          current_step: int, can_pause: bool = True,
                          can_resume: bool = False, can_step: bool = True,
                          awaiting_user_action: bool = False):
        """Update step control settings for the session."""
        session = self.load_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for step control update")
            return

        session.step_control = StepControl(
            execution_mode=execution_mode,
            current_step=current_step,
            total_steps=session.step_control.total_steps,
            can_pause=can_pause,
            can_resume=can_resume,
            can_step=can_step,
            awaiting_user_action=awaiting_user_action
        )

        self.save_session(session)

        logger.info("🎮 Step Control Updated",
                   session_id=session_id,
                   mode=execution_mode.value,
                   step=f"{current_step}/{session.step_control.total_steps}",
                   awaiting=awaiting_user_action)

    def pause_session(self, session_id: str) -> bool:
        """Pause session execution."""
        session = self.load_session(session_id)
        if not session:
            return False

        if session.step_control.can_pause:
            session.current_phase = SessionPhase.PAUSED
            session.step_control.execution_mode = ExecutionMode.PAUSED
            session.step_control.can_resume = True
            session.step_control.can_pause = False
            session.step_control.awaiting_user_action = True

            self.add_progress_message(
                session_id=session_id,
                phase=SessionPhase.PAUSED,
                title="⏸️ Generation Paused",
                description="Asset generation has been paused. You can resume or step through the process.",
                progress_percent=session.completion_percentage
            )

            return True
        return False

    def resume_session(self, session_id: str) -> bool:
        """Resume paused session execution."""
        session = self.load_session(session_id)
        if not session:
            return False

        if session.step_control.can_resume:
            # Restore previous phase (not paused)
            last_non_pause_message = None
            for msg in reversed(session.progress_messages):
                if msg.phase != SessionPhase.PAUSED:
                    last_non_pause_message = msg
                    break

            if last_non_pause_message:
                session.current_phase = last_non_pause_message.phase
            else:
                session.current_phase = SessionPhase.INITIALIZATION

            session.step_control.execution_mode = ExecutionMode.CONTINUOUS
            session.step_control.can_resume = False
            session.step_control.can_pause = True
            session.step_control.awaiting_user_action = False

            self.add_progress_message(
                session_id=session_id,
                phase=session.current_phase,
                title="▶️ Generation Resumed",
                description="Asset generation has resumed and will continue automatically.",
                progress_percent=session.completion_percentage
            )

            return True
        return False

    def step_session(self, session_id: str) -> bool:
        """Execute one step then pause."""
        session = self.load_session(session_id)
        if not session:
            return False

        if session.step_control.can_step:
            session.step_control.execution_mode = ExecutionMode.STEPPING
            session.step_control.current_step += 1
            session.step_control.awaiting_user_action = False

            self.add_progress_message(
                session_id=session_id,
                phase=SessionPhase.STEPPING,
                title="👣 Stepping Through Generation",
                description=f"Executing step {session.step_control.current_step} of {session.step_control.total_steps}",
                progress_percent=session.completion_percentage
            )

            return True
        return False

    def get_progress_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent progress messages for a session."""
        session = self.load_session(session_id)
        if not session:
            return []

        # Get the most recent messages
        recent_messages = session.progress_messages[-limit:] if session.progress_messages else []
        return [msg.to_dict() for msg in recent_messages]

    def get_step_control_status(self, session_id: str) -> dict[str, Any] | None:
        """Get current step control status for a session."""
        session = self.load_session(session_id)
        if not session:
            return None

        return session.step_control.to_dict()

    def create_detailed_progress_messages(self) -> dict[SessionPhase, dict[str, str]]:
        """Create a mapping of detailed progress messages for each phase."""
        return {
            SessionPhase.INITIALIZATION: {
                "title": "🚀 Initializing Generation",
                "description": "Setting up the 3D asset generation pipeline and validating your requirements..."
            },
            SessionPhase.PLANNING: {
                "title": "🗓️ Planning Asset Structure",
                "description": "Breaking down your request into specific 3D modeling tasks and determining the best approach..."  # noqa: E501
            },
            SessionPhase.RETRIEVAL: {
                "title": "📚 Gathering Documentation",
                "description": "Retrieving relevant Blender API documentation and 3D modeling techniques for your asset..."  # noqa: E501
            },
            SessionPhase.CODE_GENERATION: {
                "title": "🔨 Generating 3D Code",
                "description": "Writing Python code to create your 3D asset using Blender's powerful modeling tools..."
            },
            SessionPhase.CODE_VALIDATION: {
                "title": "🔍 Validating Generated Code",
                "description": "Checking code syntax, imports, and Blender API usage to ensure everything is correct..."
            },
            SessionPhase.CODE_HEALING: {
                "title": "🩹 Refining Code Quality",
                "description": "Automatically fixing any issues found and optimizing the code for better performance..."
            },
            SessionPhase.EXECUTION: {
                "title": "⚙️ Executing in Blender",
                "description": "Running your 3D generation code in Blender to create the actual asset..."
            },
            SessionPhase.QUALITY_ASSESSMENT: {
                "title": "🎯 Assessing Quality",
                "description": "Analyzing the generated asset to ensure it meets your requirements and quality standards..."  # noqa: E501
            },
            SessionPhase.REFINEMENT: {
                "title": "✨ Final Refinements",
                "description": "Making final improvements and optimizations to perfect your 3D asset..."
            },
            SessionPhase.COMPLETION: {
                "title": "🎉 Generation Complete!",
                "description": "Your 3D asset has been successfully generated and is ready for download!"
            },
            SessionPhase.ERROR: {
                "title": "⚠️ Issue Encountered",
                "description": "An issue was encountered during generation. Don't worry - we're working to resolve it..."  # noqa: E501
            },
            SessionPhase.PAUSED: {
                "title": "⏸️ Generation Paused",
                "description": "Generation is paused. Use the controls below to resume or step through the process."
            },
            SessionPhase.STEPPING: {
                "title": "👣 Stepping Through",
                "description": "Executing one step at a time. Click 'Next Step' to continue or 'Resume' for automatic execution."  # noqa: E501
            }
        }

    # Enhanced granular progress tracking methods

    def add_substep_progress(self, session_id: str, parent_phase: SessionPhase,
                           substep_name: str, status: str = 'pending',
                           progress_percent: float = 0.0,
                           technical_details: dict[str, Any] | None = None) -> str:
        """Add a granular substep within a phase."""
        session = self.load_session(session_id)
        if not session:
            return ""

        substep_id = f"substep_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        now = datetime.now(UTC)

        substep = SubstepProgress(
            substep_id=substep_id,
            name=substep_name,
            status=status,
            progress_percent=progress_percent,
            start_time=now if status == 'in_progress' else None,
            technical_details=technical_details or {}
        )

        # Find the latest progress message for this phase and add substep
        for msg in reversed(session.progress_messages):
            if msg.phase == parent_phase:
                if not msg.substeps:
                    msg.substeps = []
                msg.substeps.append(substep)
                break
        else:
            # Create a new progress message with this substep
            self.add_progress_message(
                session_id=session_id,
                phase=parent_phase,
                title=f"Working on {substep_name}",
                description=f"Processing substep: {substep_name}",
                progress_percent=progress_percent,
                substeps=[substep],
                technical_details=technical_details
            )

        self.save_session(session)
        logger.debug(f"Added substep {substep_id} to session {session_id}",
                    substep_name=substep_name, status=status)

        return substep_id

    def update_substep_progress(self, session_id: str, substep_id: str,
                              status: str, progress_percent: float = None,
                              technical_details: dict[str, Any] | None = None):
        """Update the progress of a specific substep."""
        session = self.load_session(session_id)
        if not session:
            return

        now = datetime.now(UTC)

        # Find and update the substep
        for msg in session.progress_messages:
            if msg.substeps:
                for substep in msg.substeps:
                    if substep.substep_id == substep_id:
                        substep.status = status
                        if progress_percent is not None:
                            substep.progress_percent = progress_percent
                        if status == 'in_progress' and not substep.start_time:
                            substep.start_time = now
                        elif status in ['completed', 'failed', 'skipped']:
                            substep.end_time = now
                        if technical_details:
                            substep.technical_details.update(technical_details)

                        self.save_session(session)
                        logger.debug(f"Updated substep {substep_id}", status=status, progress=progress_percent)
                        return

    def create_interaction_point(self, session_id: str, phase: SessionPhase,
                               question: str, interaction_type: str = 'user_input',
                               options: list[str] | None = None,
                               timeout_seconds: int | None = None) -> str:
        """Create an interaction point where user input is needed."""
        session = self.load_session(session_id)
        if not session:
            return ""

        interaction_id = f"interaction_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        now = datetime.now(UTC)

        interaction_point = InteractionPoint(
            interaction_id=interaction_id,
            phase=phase,
            question=question,
            interaction_type=interaction_type,
            options=options or [],
            created_at=now,
            timeout_seconds=timeout_seconds,
            is_resolved=False
        )

        # Pause the session for user input
        session.step_control.execution_mode = ExecutionMode.PAUSED
        session.step_control.awaiting_user_action = True
        session.step_control.can_resume = True
        session.current_phase = SessionPhase.PAUSED

        # Add interaction point to the latest progress message
        for msg in reversed(session.progress_messages):
            if msg.phase == phase:
                if not msg.interaction_points:
                    msg.interaction_points = []
                msg.interaction_points.append(interaction_point)
                break
        else:
            # Create new progress message with interaction
            self.add_progress_message(
                session_id=session_id,
                phase=SessionPhase.PAUSED,
                title="🤔 Need Your Input",
                description=question,
                progress_percent=session.completion_percentage,
                interaction_points=[interaction_point]
            )

        self.save_session(session)
        logger.info(f"Created interaction point {interaction_id}", question=question, type=interaction_type)

        return interaction_id

    def resolve_interaction_point(self, session_id: str, interaction_id: str,
                                response: str) -> bool:
        """Resolve a user interaction point with their response."""
        session = self.load_session(session_id)
        if not session:
            return False

        now = datetime.now(UTC)

        # Find and resolve the interaction point
        for msg in session.progress_messages:
            if msg.interaction_points:
                for interaction in msg.interaction_points:
                    if interaction.interaction_id == interaction_id and not interaction.is_resolved:
                        interaction.is_resolved = True
                        interaction.response = response
                        interaction.resolved_at = now

                        # Resume the session
                        session.step_control.awaiting_user_action = False
                        session.step_control.execution_mode = ExecutionMode.CONTINUOUS

                        # Add confirmation message
                        self.add_progress_message(
                            session_id=session_id,
                            phase=session.current_phase,
                            title="✅ Input Received",
                            description=f"Thank you for your input: {response}",
                            progress_percent=session.completion_percentage
                        )

                        self.save_session(session)
                        logger.info(f"Resolved interaction {interaction_id}", response=response)
                        return True

        return False

    def add_agent_reflection(self, session_id: str, phase: SessionPhase,
                           agent_name: str, reflection_text: str,
                           confidence_score: float, decision_made: str,
                           reasoning: str = "", alternative_approaches: list[str] | None = None):
        """Add agent reflection with confidence scoring."""
        session = self.load_session(session_id)
        if not session:
            return

        reflection_id = f"reflection_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        now = datetime.now(UTC)

        reflection = AgentReflection(
            reflection_id=reflection_id,
            agent_name=agent_name,
            phase=phase,
            timestamp=now,
            reflection_text=reflection_text,
            confidence_score=max(0.0, min(1.0, confidence_score)),  # Clamp to [0,1]
            decision_made=decision_made,
            reasoning=reasoning,
            alternative_approaches=alternative_approaches or []
        )

        # Add reflection to the latest progress message for this phase
        for msg in reversed(session.progress_messages):
            if msg.phase == phase:
                if not msg.agent_reflections:
                    msg.agent_reflections = []
                msg.agent_reflections.append(reflection)

                # Update message description to include confidence info
                confidence_indicator = "🎯" if confidence_score >= 0.8 else "🤔" if confidence_score >= 0.5 else "⚠️"
                msg.description += f" {confidence_indicator} Agent confidence: {confidence_score:.0%}"
                break

        self.save_session(session)
        logger.debug(f"Added agent reflection {reflection_id}",
                    agent=agent_name, confidence=confidence_score, decision=decision_made)

    def get_enhanced_progress_summary(self, session_id: str) -> dict[str, Any] | None:  # noqa: PLR0912
        """Get a comprehensive progress summary with all granular details."""
        session = self.load_session(session_id)
        if not session:
            return None

        summary = {
            "session_id": session_id,
            "current_phase": session.current_phase.value,
            "completion_percentage": session.completion_percentage,
            "step_control": session.step_control.to_dict(),
            "total_messages": len(session.progress_messages),
            "phases_completed": [],
            "active_substeps": [],
            "pending_interactions": [],
            "agent_reflections_summary": {},
            "estimated_completion": None
        }

        # Analyze phases and substeps
        phase_status = {}
        total_substeps = 0
        completed_substeps = 0

        for msg in session.progress_messages:
            phase_name = msg.phase.value
            if phase_name not in phase_status:
                phase_status[phase_name] = {"messages": 0, "substeps": 0, "completed_substeps": 0}

            phase_status[phase_name]["messages"] += 1

            if msg.substeps:
                for substep in msg.substeps:
                    total_substeps += 1
                    phase_status[phase_name]["substeps"] += 1

                    if substep.status == 'completed':
                        completed_substeps += 1
                        phase_status[phase_name]["completed_substeps"] += 1
                    elif substep.status == 'in_progress':
                        summary["active_substeps"].append({
                            "phase": phase_name,
                            "name": substep.name,
                            "progress": substep.progress_percent
                        })

            # Check for pending interactions
            if msg.interaction_points:
                for interaction in msg.interaction_points:
                    if not interaction.is_resolved:
                        summary["pending_interactions"].append({
                            "id": interaction.interaction_id,
                            "question": interaction.question,
                            "type": interaction.interaction_type,
                            "options": interaction.options
                        })

            # Summarize agent reflections
            if msg.agent_reflections:
                for reflection in msg.agent_reflections:
                    agent_name = reflection.agent_name
                    if agent_name not in summary["agent_reflections_summary"]:
                        summary["agent_reflections_summary"][agent_name] = {
                            "total_reflections": 0,
                            "average_confidence": 0.0,
                            "decisions": []
                        }

                    agent_summary = summary["agent_reflections_summary"][agent_name]
                    agent_summary["total_reflections"] += 1
                    agent_summary["average_confidence"] = (
                        (agent_summary["average_confidence"] * (agent_summary["total_reflections"] - 1) +
                         reflection.confidence_score) / agent_summary["total_reflections"]
                    )
                    agent_summary["decisions"].append(reflection.decision_made)

        summary["phase_breakdown"] = phase_status
        summary["substep_completion_rate"] = completed_substeps / total_substeps if total_substeps > 0 else 1.0

        # Estimate completion time based on progress
        if session.completion_percentage > 0:
            elapsed_time = (datetime.now(UTC) - session.created_at).total_seconds()
            estimated_total_time = elapsed_time / (session.completion_percentage / 100.0)
            remaining_time = estimated_total_time - elapsed_time
            summary["estimated_completion"] = max(0, remaining_time)

        return summary
