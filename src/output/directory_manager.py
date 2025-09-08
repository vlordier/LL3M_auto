"""Enhanced directory management for organized 3D asset generation outputs."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class OutputDirectoryManager:
    """Manages sophisticated directory structure for 3D asset generation outputs."""
    
    def __init__(self, base_output_dir: Path = Path("outputs")):
        """Initialize the directory manager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_output_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard directory structure
        self.projects_dir = self.base_dir / "projects"
        self.sessions_dir = self.base_dir / "sessions" 
        self.gallery_dir = self.base_dir / "gallery"
        self.templates_dir = self.base_dir / "templates"
        self.refinements_dir = self.base_dir / "refinements"
        
        self._setup_base_structure()
        
    def _setup_base_structure(self):
        """Create the base directory structure."""
        directories = [
            self.projects_dir,
            self.sessions_dir,
            self.gallery_dir,
            self.templates_dir,
            self.refinements_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create gallery subdirectories
        for category in ["featured", "recent", "favorites", "archived"]:
            (self.gallery_dir / category).mkdir(exist_ok=True)
            
    def create_project(self, name: str, description: str = "") -> Path:
        """Create a new project directory.
        
        Args:
            name: Project name (will be sanitized)
            description: Project description
            
        Returns:
            Path to the project directory
        """
        # Sanitize project name
        safe_name = self._sanitize_name(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{safe_name}_{timestamp}"
        
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        subdirs = [
            "assets",          # Generated 3D assets
            "iterations",      # Refinement iterations
            "exports",         # Different file format exports
            "renders",         # Screenshots and renders
            "code",           # Generated scripts
            "metadata",       # JSON metadata files
            "user_feedback",  # User instructions and feedback
            "backups",        # Asset backups
        ]
        
        for subdir in subdirs:
            (project_dir / subdir).mkdir(exist_ok=True)
            
        # Create project metadata
        metadata = {
            "project_id": project_id,
            "name": name,
            "safe_name": safe_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0.0",
            "total_assets": 0,
            "total_iterations": 0,
            "status": "active",
            "tags": [],
        }
        
        self._save_metadata(project_dir / "metadata" / "project.json", metadata)
        
        logger.info(f"Created project: {project_id}", project_dir=str(project_dir))
        return project_dir
        
    def create_session(self, project_dir: Path) -> Path:
        """Create a new generation session within a project.
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Path to the session directory
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_dir = project_dir / "assets" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session structure
        subdirs = ["raw", "refined", "final", "variants", "experiments"]
        for subdir in subdirs:
            (session_dir / subdir).mkdir(exist_ok=True)
            
        # Session metadata
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "generation_count": 0,
            "refinement_count": 0,
            "user_satisfaction": None,
        }
        
        self._save_metadata(session_dir / "session_metadata.json", metadata)
        
        logger.info(f"Created session: {session_id}", session_dir=str(session_dir))
        return session_dir
        
    def create_asset_directory(
        self, 
        session_dir: Path, 
        asset_name: str,
        complexity_level: int = 1,
        iteration: int = 0,
        stage: str = "raw"
    ) -> Path:
        """Create a directory for a specific asset.
        
        Args:
            session_dir: Session directory
            asset_name: Name of the asset
            complexity_level: Complexity level (1-5)
            iteration: Iteration number for refinements
            stage: Asset stage (raw, refined, final, variants, experiments)
            
        Returns:
            Path to asset directory
        """
        # Create structured asset name
        safe_name = self._sanitize_name(asset_name)
        timestamp = datetime.now().strftime("%H%M%S")
        
        if iteration > 0:
            asset_id = f"{safe_name}_L{complexity_level}_iter{iteration:02d}_{timestamp}"
        else:
            asset_id = f"{safe_name}_L{complexity_level}_{timestamp}"
            
        asset_dir = session_dir / stage / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create asset structure
        subdirs = [
            "source",         # Generated code and scripts
            "outputs",        # Blender outputs (.blend, renders)
            "exports",        # Multi-format exports
            "analysis",       # Critique and quality analysis
            "user_notes",     # User feedback and instructions
            "history",        # Change history
        ]
        
        for subdir in subdirs:
            (asset_dir / subdir).mkdir(exist_ok=True)
            
        # Asset metadata
        metadata = {
            "asset_id": asset_id,
            "asset_name": asset_name,
            "safe_name": safe_name,
            "complexity_level": complexity_level,
            "iteration": iteration,
            "stage": stage,
            "created_at": datetime.now().isoformat(),
            "status": "generating",
            "generation_time": None,
            "quality_score": None,
            "user_rating": None,
            "tags": [f"level-{complexity_level}", stage],
        }
        
        self._save_metadata(asset_dir / "asset_metadata.json", metadata)
        
        logger.info(f"Created asset directory: {asset_id}", asset_dir=str(asset_dir))
        return asset_dir
        
    def get_recent_projects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent projects with metadata.
        
        Args:
            limit: Maximum number of projects to return
            
        Returns:
            List of project metadata dictionaries
        """
        projects = []
        
        for project_dir in sorted(self.projects_dir.iterdir(), 
                                key=lambda p: p.stat().st_mtime, reverse=True):
            if project_dir.is_dir():
                metadata_file = project_dir / "metadata" / "project.json"
                if metadata_file.exists():
                    metadata = self._load_metadata(metadata_file)
                    metadata["path"] = str(project_dir)
                    projects.append(metadata)
                    
                if len(projects) >= limit:
                    break
                    
        return projects
        
    def find_assets(
        self,
        project_dir: Optional[Path] = None,
        complexity_level: Optional[int] = None,
        stage: Optional[str] = None,
        min_quality: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find assets matching criteria.
        
        Args:
            project_dir: Specific project to search in
            complexity_level: Filter by complexity level
            stage: Filter by stage
            min_quality: Minimum quality score
            tags: Required tags
            
        Returns:
            List of matching asset metadata
        """
        assets = []
        
        search_dirs = [project_dir] if project_dir else list(self.projects_dir.iterdir())
        
        for proj_dir in search_dirs:
            if not proj_dir.is_dir():
                continue
                
            assets_dir = proj_dir / "assets"
            if not assets_dir.exists():
                continue
                
            # Search through sessions
            for session_dir in assets_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                    
                # Search through stages
                for stage_dir in session_dir.iterdir():
                    if not stage_dir.is_dir():
                        continue
                        
                    # Search through assets
                    for asset_dir in stage_dir.iterdir():
                        if not asset_dir.is_dir():
                            continue
                            
                        metadata_file = asset_dir / "asset_metadata.json"
                        if not metadata_file.exists():
                            continue
                            
                        metadata = self._load_metadata(metadata_file)
                        
                        # Apply filters
                        if complexity_level and metadata.get("complexity_level") != complexity_level:
                            continue
                        if stage and metadata.get("stage") != stage:
                            continue
                        if min_quality and (metadata.get("quality_score") or 0) < min_quality:
                            continue
                        if tags:
                            asset_tags = metadata.get("tags", [])
                            if not all(tag in asset_tags for tag in tags):
                                continue
                                
                        metadata["path"] = str(asset_dir)
                        assets.append(metadata)
                        
        # Sort by creation time (newest first)
        assets.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        return assets
        
    def archive_project(self, project_dir: Path):
        """Archive a completed project.
        
        Args:
            project_dir: Project directory to archive
        """
        # Update project metadata
        metadata_file = project_dir / "metadata" / "project.json"
        if metadata_file.exists():
            metadata = self._load_metadata(metadata_file)
            metadata["status"] = "archived"
            metadata["archived_at"] = datetime.now().isoformat()
            self._save_metadata(metadata_file, metadata)
            
        logger.info(f"Archived project: {project_dir.name}")
        
    def promote_to_gallery(self, asset_dir: Path, category: str = "featured"):
        """Promote an asset to the gallery.
        
        Args:
            asset_dir: Asset directory to promote
            category: Gallery category (featured, recent, favorites)
        """
        # Copy asset to gallery with symlink
        metadata_file = asset_dir / "asset_metadata.json"
        if metadata_file.exists():
            metadata = self._load_metadata(metadata_file)
            asset_id = metadata["asset_id"]
            
            gallery_link = self.gallery_dir / category / asset_id
            if not gallery_link.exists():
                gallery_link.symlink_to(asset_dir)
                
            # Update metadata
            metadata["gallery_categories"] = metadata.get("gallery_categories", [])
            if category not in metadata["gallery_categories"]:
                metadata["gallery_categories"].append(category)
                metadata["promoted_at"] = datetime.now().isoformat()
                self._save_metadata(metadata_file, metadata)
                
            logger.info(f"Promoted asset {asset_id} to gallery: {category}")
            
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old temporary files and unsuccessful generations.
        
        Args:
            days_old: Delete files older than this many days
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        # Clean up failed generations and temporary files
        for root in [self.sessions_dir, self.base_dir / "temp"]:
            if not root.exists():
                continue
                
            for file_path in root.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        # Only delete if it looks like a temp/failed file
                        if any(marker in file_path.name.lower() 
                              for marker in ["temp", "tmp", "failed", "error", ".lock"]):
                            file_path.unlink()
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Could not clean up {file_path}: {e}")
                        
        logger.info(f"Cleaned up {cleaned_count} old files")
        
    def get_directory_stats(self) -> Dict[str, Any]:
        """Get statistics about the directory structure.
        
        Returns:
            Dictionary with various statistics
        """
        stats = {
            "total_projects": len(list(self.projects_dir.iterdir())),
            "total_size_mb": self._get_directory_size(self.base_dir) / (1024 * 1024),
            "recent_activity": self._get_recent_activity(),
            "top_complexity_levels": self._get_complexity_stats(),
            "gallery_counts": {
                cat.name: len(list(cat.iterdir()))
                for cat in self.gallery_dir.iterdir()
                if cat.is_dir()
            }
        }
        
        return stats
        
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in file paths."""
        import re
        # Remove or replace problematic characters
        safe = re.sub(r'[^\w\-_\.]', '_', name)
        safe = re.sub(r'_{2,}', '_', safe)  # Replace multiple underscores
        return safe.strip('_')[:50]  # Limit length
        
    def _save_metadata(self, file_path: Path, metadata: Dict[str, Any]):
        """Save metadata to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load metadata from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata from {file_path}: {e}")
            return {}
            
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not calculate size for {directory}: {e}")
        return total
        
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity across all projects."""
        activity = []
        
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
                
            # Check for recent asset creation
            assets_dir = project_dir / "assets"
            if assets_dir.exists():
                for session_dir in assets_dir.iterdir():
                    if session_dir.is_dir():
                        metadata_file = session_dir / "session_metadata.json"
                        if metadata_file.exists():
                            metadata = self._load_metadata(metadata_file)
                            activity.append({
                                "type": "session",
                                "project": project_dir.name,
                                "session": session_dir.name,
                                "timestamp": metadata.get("created_at"),
                            })
                            
        # Sort by timestamp and return recent ones
        activity.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        return activity[:10]
        
    def _get_complexity_stats(self) -> Dict[int, int]:
        """Get statistics about complexity levels."""
        stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        assets = self.find_assets()
        for asset in assets:
            level = asset.get("complexity_level", 1)
            if level in stats:
                stats[level] += 1
                
        return stats