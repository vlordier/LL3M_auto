"""Asset management system for LL3M-generated 3D assets."""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from ..utils.types import AssetMetadata, SubTask

logger = structlog.get_logger(__name__)


class AssetVersion(BaseModel):
    """Represents a version of an asset."""

    version: int = Field(..., ge=1, description="Version number")
    timestamp: float = Field(..., description="Creation timestamp")
    file_path: str = Field(..., description="Path to asset file")
    screenshot_path: Optional[str] = Field(None, description="Path to screenshot")
    refinement_request: Optional[str] = Field(
        None, description="Refinement that led to this version"
    )
    quality_score: Optional[float] = Field(
        None, ge=0, le=10, description="Quality score"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ManagedAsset(BaseModel):
    """Represents a managed asset with versioning."""

    id: str = Field(..., description="Unique asset identifier")
    name: str = Field(..., description="Human-readable asset name")
    original_prompt: str = Field(..., description="Original creation prompt")
    subtasks: list[SubTask] = Field(
        default_factory=list, description="Subtasks used to create asset"
    )
    created_at: float = Field(..., description="Creation timestamp")
    updated_at: float = Field(..., description="Last update timestamp")
    versions: list[AssetVersion] = Field(
        default_factory=list, description="Asset versions"
    )
    tags: list[str] = Field(default_factory=list, description="Asset tags")
    current_version: int = Field(default=1, ge=1, description="Current active version")

    @property
    def latest_version(self) -> Optional[AssetVersion]:
        """Get the latest version of the asset."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version)

    @property
    def current_asset_version(self) -> Optional[AssetVersion]:
        """Get the current active version of the asset."""
        for version in self.versions:
            if version.version == self.current_version:
                return version
        return None


class AssetRepository:
    """Repository for storing and managing assets."""

    def __init__(self, base_path: str = "assets"):
        """Initialize asset repository."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Create subdirectories
        self.assets_dir = self.base_path / "files"
        self.metadata_dir = self.base_path / "metadata"
        self.screenshots_dir = self.base_path / "screenshots"

        for directory in [self.assets_dir, self.metadata_dir, self.screenshots_dir]:
            directory.mkdir(exist_ok=True)

        self._assets_cache: dict[str, ManagedAsset] = {}
        self._load_assets_index()

        logger.info("Asset repository initialized", base_path=str(self.base_path))

    def _load_assets_index(self) -> None:
        """Load assets index from metadata files."""
        self._assets_cache.clear()

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    asset_data = json.load(f)
                    asset = ManagedAsset(**asset_data)
                    self._assets_cache[asset.id] = asset

            except Exception as e:
                logger.error(
                    "Failed to load asset metadata",
                    file=str(metadata_file),
                    error=str(e),
                )

        logger.info("Loaded assets index", count=len(self._assets_cache))

    def create_asset(
        self,
        asset_metadata: AssetMetadata,
        quality_score: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> ManagedAsset:
        """Create a new managed asset."""
        asset_id = asset_metadata.id
        current_time = time.time()

        if tags is None:
            tags = []

        # Generate asset name from prompt
        asset_name = self._generate_asset_name(asset_metadata.prompt)

        # Create managed asset
        managed_asset = ManagedAsset(
            id=asset_id,
            name=asset_name,
            original_prompt=asset_metadata.prompt,
            subtasks=asset_metadata.subtasks or [],
            created_at=current_time,
            updated_at=current_time,
            tags=tags,
            versions=[],
            current_version=1,
        )

        # Add initial version
        version = AssetVersion(
            version=1,
            timestamp=current_time,
            file_path=asset_metadata.file_path or "",
            screenshot_path=asset_metadata.screenshot_path,
            refinement_request=None,
            quality_score=quality_score,
            metadata={"initial_creation": True},
        )

        managed_asset.versions.append(version)

        # Store asset files in repository
        self._store_asset_files(managed_asset, version, asset_metadata)

        # Save metadata
        self._save_asset_metadata(managed_asset)

        # Update cache
        self._assets_cache[asset_id] = managed_asset

        logger.info(
            "Created managed asset",
            asset_id=asset_id,
            name=asset_name,
            quality_score=quality_score,
        )

        return managed_asset

    def add_version(
        self,
        asset_id: str,
        asset_metadata: AssetMetadata,
        refinement_request: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> Optional[AssetVersion]:
        """Add a new version to an existing asset."""
        if asset_id not in self._assets_cache:
            logger.error("Asset not found", asset_id=asset_id)
            return None

        managed_asset = self._assets_cache[asset_id]
        next_version = max(v.version for v in managed_asset.versions) + 1
        current_time = time.time()

        # Create new version
        version = AssetVersion(
            version=next_version,
            timestamp=current_time,
            file_path=asset_metadata.file_path or "",
            screenshot_path=asset_metadata.screenshot_path,
            refinement_request=refinement_request,
            quality_score=quality_score,
            metadata={"refinement": True},
        )

        # Store asset files
        self._store_asset_files(managed_asset, version, asset_metadata)

        # Add version to asset
        managed_asset.versions.append(version)
        managed_asset.current_version = next_version
        managed_asset.updated_at = current_time

        # Save metadata
        self._save_asset_metadata(managed_asset)

        logger.info(
            "Added asset version",
            asset_id=asset_id,
            version=next_version,
            quality_score=quality_score,
        )

        return version

    def get_asset(self, asset_id: str) -> Optional[ManagedAsset]:
        """Get a managed asset by ID."""
        return self._assets_cache.get(asset_id)

    def list_assets(
        self,
        tags: Optional[list[str]] = None,
        min_quality_score: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[ManagedAsset]:
        """List assets with optional filtering."""
        assets = list(self._assets_cache.values())

        # Filter by tags
        if tags:
            assets = [
                asset for asset in assets if any(tag in asset.tags for tag in tags)
            ]

        # Filter by quality score
        if min_quality_score is not None:
            filtered_assets = []
            for asset in assets:
                latest_version = asset.latest_version
                if (
                    latest_version
                    and latest_version.quality_score
                    and latest_version.quality_score >= min_quality_score
                ):
                    filtered_assets.append(asset)
            assets = filtered_assets

        # Sort by updated_at (most recent first)
        assets.sort(key=lambda a: a.updated_at, reverse=True)

        # Apply limit
        if limit:
            assets = assets[:limit]

        return assets

    def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset and all its versions."""
        if asset_id not in self._assets_cache:
            return False

        managed_asset = self._assets_cache[asset_id]

        # Delete all version files
        for version in managed_asset.versions:
            self._delete_version_files(managed_asset, version)

        # Delete metadata file
        metadata_file = self.metadata_dir / f"{asset_id}.json"
        metadata_file.unlink(missing_ok=True)

        # Remove from cache
        del self._assets_cache[asset_id]

        logger.info("Deleted asset", asset_id=asset_id)
        return True

    def delete_version(self, asset_id: str, version_number: int) -> bool:
        """Delete a specific version of an asset."""
        if asset_id not in self._assets_cache:
            return False

        managed_asset = self._assets_cache[asset_id]

        # Find and remove version
        version_to_delete = None
        for i, version in enumerate(managed_asset.versions):
            if version.version == version_number:
                version_to_delete = version
                managed_asset.versions.pop(i)
                break

        if not version_to_delete:
            return False

        # Delete version files
        self._delete_version_files(managed_asset, version_to_delete)

        # Update current version if necessary
        if managed_asset.current_version == version_number:
            if managed_asset.versions:
                managed_asset.current_version = max(
                    v.version for v in managed_asset.versions
                )
            else:
                # No versions left, delete entire asset
                return self.delete_asset(asset_id)

        # Update timestamp
        managed_asset.updated_at = time.time()

        # Save metadata
        self._save_asset_metadata(managed_asset)

        logger.info("Deleted asset version", asset_id=asset_id, version=version_number)
        return True

    def set_current_version(self, asset_id: str, version_number: int) -> bool:
        """Set the current active version of an asset."""
        if asset_id not in self._assets_cache:
            return False

        managed_asset = self._assets_cache[asset_id]

        # Check if version exists
        if not any(v.version == version_number for v in managed_asset.versions):
            return False

        managed_asset.current_version = version_number
        managed_asset.updated_at = time.time()

        # Save metadata
        self._save_asset_metadata(managed_asset)

        logger.info(
            "Set current asset version", asset_id=asset_id, version=version_number
        )
        return True

    def get_asset_statistics(self) -> dict[str, Any]:
        """Get repository statistics."""
        assets = list(self._assets_cache.values())

        total_versions = sum(len(asset.versions) for asset in assets)
        total_size_mb = self._calculate_total_size()

        # Quality statistics
        quality_scores = []
        for asset in assets:
            latest = asset.latest_version
            if latest and latest.quality_score:
                quality_scores.append(latest.quality_score)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Tag statistics
        all_tags = []
        for asset in assets:
            all_tags.extend(asset.tags)

        tag_counts: dict[str, int] = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_assets": len(assets),
            "total_versions": total_versions,
            "total_size_mb": total_size_mb,
            "average_quality_score": avg_quality,
            "assets_with_quality_scores": len(quality_scores),
            "tag_counts": tag_counts,
            "repository_path": str(self.base_path),
        }

    def _generate_asset_name(self, prompt: str) -> str:
        """Generate a human-readable asset name from prompt."""
        # Take words and clean them, skipping empty ones
        words = prompt.split()
        clean_words: list[str] = []

        for word in words:
            # Remove non-alphanumeric characters
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word and len(clean_words) < 4:  # Limit to 4 meaningful words
                clean_words.append(clean_word.capitalize())

        if not clean_words:
            return "UnnamedAsset"

        return "_".join(clean_words)

    def _store_asset_files(
        self,
        managed_asset: ManagedAsset,
        version: AssetVersion,
        asset_metadata: AssetMetadata,
    ) -> None:
        """Store asset files in the repository."""
        # Create asset directory
        asset_dir = self.assets_dir / managed_asset.id
        asset_dir.mkdir(exist_ok=True)

        # Store main asset file
        if asset_metadata.file_path and Path(asset_metadata.file_path).exists():
            source_file = Path(asset_metadata.file_path)
            target_file = asset_dir / f"v{version.version}_{source_file.name}"

            shutil.copy2(source_file, target_file)
            version.file_path = str(target_file)

        # Store screenshot
        if (
            asset_metadata.screenshot_path
            and Path(asset_metadata.screenshot_path).exists()
        ):
            source_screenshot = Path(asset_metadata.screenshot_path)
            target_screenshot = (
                self.screenshots_dir / f"{managed_asset.id}_v{version.version}.png"
            )

            shutil.copy2(source_screenshot, target_screenshot)
            version.screenshot_path = str(target_screenshot)

    def _delete_version_files(
        self, _managed_asset: ManagedAsset, version: AssetVersion
    ) -> None:
        """Delete files for a specific version."""
        # Delete asset file
        if version.file_path:
            Path(version.file_path).unlink(missing_ok=True)

        # Delete screenshot
        if version.screenshot_path:
            Path(version.screenshot_path).unlink(missing_ok=True)

    def _save_asset_metadata(self, managed_asset: ManagedAsset) -> None:
        """Save asset metadata to disk."""
        metadata_file = self.metadata_dir / f"{managed_asset.id}.json"

        with open(metadata_file, "w") as f:
            json.dump(managed_asset.model_dump(), f, indent=2, default=str)

    def _calculate_total_size(self) -> float:
        """Calculate total size of all assets in MB."""
        total_size = 0

        try:
            for asset_dir in self.assets_dir.iterdir():
                if asset_dir.is_dir():
                    for file_path in asset_dir.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size

            # Add screenshots
            for screenshot in self.screenshots_dir.iterdir():
                if screenshot.is_file():
                    total_size += screenshot.stat().st_size

        except Exception as e:
            logger.error("Failed to calculate total size", error=str(e))

        return total_size / (1024 * 1024)  # Convert to MB


class AssetManager:
    """High-level asset management interface."""

    def __init__(self, repository_path: str = "assets"):
        """Initialize asset manager."""
        self.repository = AssetRepository(repository_path)
        logger.info("Asset manager initialized")

    def create_from_workflow_state(
        self, workflow_state: Any, tags: Optional[list[str]] = None
    ) -> Optional[ManagedAsset]:
        """Create asset from workflow state."""
        if not hasattr(workflow_state, "asset_metadata"):
            logger.error("Workflow state has no asset metadata")
            return None

        # Extract quality score from verification result
        quality_score = None
        if hasattr(workflow_state, "verification_result"):
            verification_result = workflow_state.verification_result
            if isinstance(verification_result, dict):
                quality_score = verification_result.get("quality_score")

        return self.repository.create_asset(
            workflow_state.asset_metadata, quality_score=quality_score, tags=tags
        )

    def add_refinement_version(
        self, asset_id: str, workflow_state: Any, refinement_request: str
    ) -> Optional[AssetVersion]:
        """Add a refinement version from workflow state."""
        if not hasattr(workflow_state, "asset_metadata"):
            logger.error("Workflow state has no asset metadata")
            return None

        # Extract quality score
        quality_score = None
        if hasattr(workflow_state, "verification_result"):
            verification_result = workflow_state.verification_result
            if isinstance(verification_result, dict):
                quality_score = verification_result.get("quality_score")

        return self.repository.add_version(
            asset_id,
            workflow_state.asset_metadata,
            refinement_request=refinement_request,
            quality_score=quality_score,
        )

    def get_best_assets(self, limit: int = 10) -> list[ManagedAsset]:
        """Get the highest quality assets."""
        return self.repository.list_assets(min_quality_score=8.0, limit=limit)

    def cleanup_low_quality_assets(self, min_quality: float = 5.0) -> int:
        """Remove assets with quality scores below threshold."""
        assets_to_delete = []

        for asset in self.repository.list_assets():
            latest_version = asset.latest_version
            if (
                latest_version
                and latest_version.quality_score
                and latest_version.quality_score < min_quality
            ):
                assets_to_delete.append(asset.id)

        deleted_count = 0
        for asset_id in assets_to_delete:
            if self.repository.delete_asset(asset_id):
                deleted_count += 1

        logger.info(
            "Cleaned up low quality assets",
            deleted=deleted_count,
            threshold=min_quality,
        )
        return deleted_count
