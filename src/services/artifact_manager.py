"""Artifact and asset management system for workflow interruption and resumption."""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import hashlib
import structlog
from dataclasses import asdict
import uuid

from ..utils.types import WorkflowState, ExecutionResult, SubTask

logger = structlog.get_logger(__name__)


class ArtifactMetadata:
    """Metadata for tracking workflow artifacts."""
    
    def __init__(self, artifact_id: str, artifact_type: str, file_path: str, 
                 created_at: datetime, workflow_id: str, subtask_id: Optional[str] = None,
                 checksum: Optional[str] = None, size_bytes: int = 0, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type  # 'blend_file', 'screenshot', 'code', 'state', 'logs'
        self.file_path = file_path
        self.created_at = created_at
        self.workflow_id = workflow_id
        self.subtask_id = subtask_id
        self.checksum = checksum
        self.size_bytes = size_bytes
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'artifact_id': self.artifact_id,
            'artifact_type': self.artifact_type,
            'file_path': self.file_path,
            'created_at': self.created_at.isoformat(),
            'workflow_id': self.workflow_id,
            'subtask_id': self.subtask_id,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactMetadata':
        """Create from dictionary."""
        return cls(
            artifact_id=data['artifact_id'],
            artifact_type=data['artifact_type'],
            file_path=data['file_path'],
            created_at=datetime.fromisoformat(data['created_at']),
            workflow_id=data['workflow_id'],
            subtask_id=data.get('subtask_id'),
            checksum=data.get('checksum'),
            size_bytes=data.get('size_bytes', 0),
            metadata=data.get('metadata', {})
        )


class WorkflowCheckpoint:
    """Represents a workflow checkpoint for resumption."""
    
    def __init__(self, checkpoint_id: str, workflow_id: str, state: WorkflowState,
                 created_at: datetime, step_name: str, artifacts: List[ArtifactMetadata],
                 execution_context: Optional[Dict[str, Any]] = None):
        self.checkpoint_id = checkpoint_id
        self.workflow_id = workflow_id
        self.state = state
        self.created_at = created_at
        self.step_name = step_name
        self.artifacts = artifacts
        self.execution_context = execution_context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'workflow_id': self.workflow_id,
            'state': asdict(self.state),
            'created_at': self.created_at.isoformat(),
            'step_name': self.step_name,
            'artifacts': [artifact.to_dict() for artifact in self.artifacts],
            'execution_context': self.execution_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        """Create from dictionary."""
        # Reconstruct WorkflowState from dict
        state_data = data['state']
        subtasks = [SubTask(**task_data) for task_data in state_data.get('subtasks', [])]
        
        # Handle ExecutionResult reconstruction
        exec_result = state_data.get('execution_result')
        if exec_result:
            execution_result = ExecutionResult(**exec_result)
        else:
            execution_result = None
            
        state = WorkflowState(
            prompt=state_data['prompt'],
            subtasks=subtasks,
            generated_code=state_data.get('generated_code', ''),
            execution_result=execution_result,
            healing_iterations=state_data.get('healing_iterations', 0),
            total_tokens_used=state_data.get('total_tokens_used', 0),
            performance_metrics=state_data.get('performance_metrics', {}),
            context=state_data.get('context', {})
        )
        
        artifacts = [ArtifactMetadata.from_dict(art) for art in data['artifacts']]
        
        return cls(
            checkpoint_id=data['checkpoint_id'],
            workflow_id=data['workflow_id'],
            state=state,
            created_at=datetime.fromisoformat(data['created_at']),
            step_name=data['step_name'],
            artifacts=artifacts,
            execution_context=data.get('execution_context', {})
        )


class ArtifactManager:
    """Manages workflow artifacts and enables interruption/resumption."""
    
    def __init__(self, base_path: Union[str, Path] = "artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.base_path / "workflows").mkdir(exist_ok=True)
        (self.base_path / "checkpoints").mkdir(exist_ok=True)
        (self.base_path / "assets").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        
        self.artifacts_registry: Dict[str, ArtifactMetadata] = {}
        self.checkpoints_registry: Dict[str, WorkflowCheckpoint] = {}
        
        # Load existing registries
        self._load_registries()
        
    def _load_registries(self) -> None:
        """Load artifact and checkpoint registries from disk."""
        try:
            # Load artifacts registry
            artifacts_file = self.base_path / "metadata" / "artifacts_registry.json"
            if artifacts_file.exists():
                with open(artifacts_file, 'r') as f:
                    data = json.load(f)
                    self.artifacts_registry = {
                        k: ArtifactMetadata.from_dict(v) for k, v in data.items()
                    }
                    
            # Load checkpoints registry
            checkpoints_file = self.base_path / "metadata" / "checkpoints_registry.json"
            if checkpoints_file.exists():
                with open(checkpoints_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints_registry = {
                        k: WorkflowCheckpoint.from_dict(v) for k, v in data.items()
                    }
                    
        except Exception as e:
            logger.warning(f"Failed to load registries: {e}")
            
    def _save_registries(self) -> None:
        """Save artifact and checkpoint registries to disk."""
        try:
            # Save artifacts registry
            artifacts_file = self.base_path / "metadata" / "artifacts_registry.json"
            with open(artifacts_file, 'w') as f:
                data = {k: v.to_dict() for k, v in self.artifacts_registry.items()}
                json.dump(data, f, indent=2)
                
            # Save checkpoints registry
            checkpoints_file = self.base_path / "metadata" / "checkpoints_registry.json"
            with open(checkpoints_file, 'w') as f:
                data = {k: v.to_dict() for k, v in self.checkpoints_registry.items()}
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save registries: {e}")
            
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
        
    async def store_artifact(self, workflow_id: str, artifact_type: str, 
                           source_path: Union[str, Path], 
                           subtask_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store an artifact and return its ID."""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact not found: {source_path}")
            
        # Generate unique artifact ID
        artifact_id = str(uuid.uuid4())
        
        # Create workflow-specific directory
        workflow_dir = self.base_path / "assets" / workflow_id
        workflow_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine target filename
        target_path = workflow_dir / f"{artifact_id}_{source_path.name}"
        
        # Copy file to artifact storage
        shutil.copy2(source_path, target_path)
        
        # Calculate checksum and size
        checksum = self._calculate_file_checksum(target_path)
        size_bytes = target_path.stat().st_size
        
        # Create artifact metadata
        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            file_path=str(target_path),
            created_at=datetime.now(timezone.utc),
            workflow_id=workflow_id,
            subtask_id=subtask_id,
            checksum=checksum,
            size_bytes=size_bytes,
            metadata=metadata
        )
        
        # Register artifact
        self.artifacts_registry[artifact_id] = artifact_metadata
        self._save_registries()
        
        logger.info(f"Stored artifact {artifact_id} of type {artifact_type}")
        return artifact_id
        
    async def create_checkpoint(self, workflow_id: str, state: WorkflowState,
                              step_name: str, execution_context: Optional[Dict[str, Any]] = None) -> str:
        """Create a workflow checkpoint."""
        checkpoint_id = str(uuid.uuid4())
        
        # Get all artifacts for this workflow
        workflow_artifacts = [
            artifact for artifact in self.artifacts_registry.values()
            if artifact.workflow_id == workflow_id
        ]
        
        # Create checkpoint
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            state=state,
            created_at=datetime.now(timezone.utc),
            step_name=step_name,
            artifacts=workflow_artifacts,
            execution_context=execution_context
        )
        
        # Save checkpoint to disk
        checkpoint_path = self.base_path / "checkpoints" / f"{checkpoint_id}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
            
        # Register checkpoint
        self.checkpoints_registry[checkpoint_id] = checkpoint
        self._save_registries()
        
        logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_id} at step {step_name}")
        return checkpoint_id
        
    async def get_latest_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Get the most recent checkpoint for a workflow."""
        workflow_checkpoints = [
            checkpoint for checkpoint in self.checkpoints_registry.values()
            if checkpoint.workflow_id == workflow_id
        ]
        
        if not workflow_checkpoints:
            return None
            
        # Return most recent checkpoint
        return max(workflow_checkpoints, key=lambda c: c.created_at)
        
    async def restore_from_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Restore a workflow from a checkpoint."""
        if checkpoint_id not in self.checkpoints_registry:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
        checkpoint = self.checkpoints_registry[checkpoint_id]
        
        # Verify all artifacts exist
        for artifact in checkpoint.artifacts:
            if not Path(artifact.file_path).exists():
                logger.warning(f"Artifact file missing: {artifact.file_path}")
                
        logger.info(f"Restored workflow from checkpoint {checkpoint_id}")
        return checkpoint
        
    async def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Get artifact metadata by ID."""
        return self.artifacts_registry.get(artifact_id)
        
    async def get_workflow_artifacts(self, workflow_id: str, 
                                   artifact_type: Optional[str] = None) -> List[ArtifactMetadata]:
        """Get all artifacts for a workflow, optionally filtered by type."""
        artifacts = [
            artifact for artifact in self.artifacts_registry.values()
            if artifact.workflow_id == workflow_id
        ]
        
        if artifact_type:
            artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
            
        return sorted(artifacts, key=lambda a: a.created_at)
        
    async def cleanup_workflow(self, workflow_id: str, keep_latest_checkpoint: bool = True) -> None:
        """Clean up artifacts and checkpoints for a workflow."""
        # Get workflow artifacts and checkpoints
        workflow_artifacts = await self.get_workflow_artifacts(workflow_id)
        workflow_checkpoints = [
            checkpoint for checkpoint in self.checkpoints_registry.values()
            if checkpoint.workflow_id == workflow_id
        ]
        
        # Keep latest checkpoint if requested
        if keep_latest_checkpoint and workflow_checkpoints:
            latest_checkpoint = max(workflow_checkpoints, key=lambda c: c.created_at)
            workflow_checkpoints = [c for c in workflow_checkpoints if c != latest_checkpoint]
            
        # Remove artifact files
        for artifact in workflow_artifacts:
            try:
                Path(artifact.file_path).unlink(missing_ok=True)
                del self.artifacts_registry[artifact.artifact_id]
            except Exception as e:
                logger.warning(f"Failed to remove artifact {artifact.artifact_id}: {e}")
                
        # Remove checkpoint files
        for checkpoint in workflow_checkpoints:
            try:
                checkpoint_path = self.base_path / "checkpoints" / f"{checkpoint.checkpoint_id}.json"
                checkpoint_path.unlink(missing_ok=True)
                del self.checkpoints_registry[checkpoint.checkpoint_id]
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_id}: {e}")
                
        # Remove workflow directory if empty
        try:
            workflow_dir = self.base_path / "assets" / workflow_id
            if workflow_dir.exists() and not any(workflow_dir.iterdir()):
                workflow_dir.rmdir()
        except Exception as e:
            logger.warning(f"Failed to remove workflow directory: {e}")
            
        self._save_registries()
        logger.info(f"Cleaned up workflow {workflow_id}")
        
    async def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get summary information about a workflow's artifacts and checkpoints."""
        artifacts = await self.get_workflow_artifacts(workflow_id)
        checkpoints = [
            checkpoint for checkpoint in self.checkpoints_registry.values()
            if checkpoint.workflow_id == workflow_id
        ]
        
        artifact_summary = {}
        for artifact in artifacts:
            artifact_summary[artifact.artifact_type] = artifact_summary.get(artifact.artifact_type, 0) + 1
            
        total_size = sum(artifact.size_bytes for artifact in artifacts)
        
        return {
            'workflow_id': workflow_id,
            'total_artifacts': len(artifacts),
            'artifact_types': artifact_summary,
            'total_size_bytes': total_size,
            'total_checkpoints': len(checkpoints),
            'latest_checkpoint': max(checkpoints, key=lambda c: c.created_at).created_at.isoformat() if checkpoints else None,
            'created_at': min(artifacts, key=lambda a: a.created_at).created_at.isoformat() if artifacts else None
        }