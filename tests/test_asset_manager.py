"""Tests for AssetManager and related components."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.assets.manager import (
    AssetManager,
    AssetRepository,
    AssetVersion,
    ManagedAsset,
)
from src.utils.types import AssetMetadata, SubTask, TaskType


@pytest.fixture
def temp_repo_path():
    """Create a temporary directory for repository testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_subtasks():
    """Sample subtasks for testing."""
    return [
        SubTask(
            id="task1",
            type=TaskType.GEOMETRY,
            description="Create a cube",
            parameters={"shape": "cube", "size": 2.0},
        ),
        SubTask(
            id="task2",
            type=TaskType.MATERIAL,
            description="Add red material",
            parameters={"color": [1.0, 0.0, 0.0], "metallic": 0.0},
        ),
    ]


@pytest.fixture
def sample_asset_metadata(sample_subtasks, tmp_path):
    """Sample asset metadata for testing."""
    # Create sample files
    asset_file = tmp_path / "test_asset.blend"
    asset_file.write_bytes(b"mock blend data")

    screenshot_file = tmp_path / "test_screenshot.png"
    screenshot_file.write_bytes(b"mock png data")

    return AssetMetadata(
        id="test_asset_001",
        prompt="Create a red cube with metallic finish",
        file_path=str(asset_file),
        screenshot_path=str(screenshot_file),
        subtasks=sample_subtasks,
    )


class TestAssetVersion:
    """Test cases for AssetVersion."""

    def test_initialization(self):
        """Test AssetVersion initialization."""
        version = AssetVersion(
            version=1,
            timestamp=time.time(),
            file_path="/path/to/asset.blend",
            screenshot_path="/path/to/screenshot.png",
            refinement_request="Make it more detailed",
            quality_score=8.5,
            metadata={"test": "data"},
        )

        assert version.version == 1
        assert version.timestamp > 0
        assert version.file_path == "/path/to/asset.blend"
        assert version.screenshot_path == "/path/to/screenshot.png"
        assert version.refinement_request == "Make it more detailed"
        assert version.quality_score == 8.5
        assert version.metadata["test"] == "data"

    def test_initialization_minimal(self):
        """Test AssetVersion with minimal required fields."""
        version = AssetVersion(
            version=1, timestamp=time.time(), file_path="/path/to/asset.blend"
        )

        assert version.version == 1
        assert version.file_path == "/path/to/asset.blend"
        assert version.screenshot_path is None
        assert version.refinement_request is None
        assert version.quality_score is None
        assert version.metadata == {}

    def test_validation_version_positive(self):
        """Test version number validation."""
        with pytest.raises(ValueError):
            AssetVersion(
                version=0,  # Should be >= 1
                timestamp=time.time(),
                file_path="/path/to/asset.blend",
            )

    def test_validation_quality_score_range(self):
        """Test quality score validation."""
        # Valid range
        version = AssetVersion(
            version=1,
            timestamp=time.time(),
            file_path="/path/to/asset.blend",
            quality_score=5.5,
        )
        assert version.quality_score == 5.5

        # Invalid range
        with pytest.raises(ValueError):
            AssetVersion(
                version=1,
                timestamp=time.time(),
                file_path="/path/to/asset.blend",
                quality_score=15.0,  # Should be <= 10
            )


class TestManagedAsset:
    """Test cases for ManagedAsset."""

    def test_initialization(self, sample_subtasks):
        """Test ManagedAsset initialization."""
        current_time = time.time()

        asset = ManagedAsset(
            id="test_001",
            name="Test_Red_Cube",
            original_prompt="Create a red cube",
            subtasks=sample_subtasks,
            created_at=current_time,
            updated_at=current_time,
            tags=["cube", "red", "basic"],
            current_version=1,
        )

        assert asset.id == "test_001"
        assert asset.name == "Test_Red_Cube"
        assert asset.original_prompt == "Create a red cube"
        assert len(asset.subtasks) == 2
        assert asset.created_at == current_time
        assert asset.updated_at == current_time
        assert asset.tags == ["cube", "red", "basic"]
        assert asset.current_version == 1
        assert asset.versions == []

    def test_latest_version_property_empty(self):
        """Test latest_version property with no versions."""
        asset = ManagedAsset(
            id="test_001",
            name="Test_Asset",
            original_prompt="Test prompt",
            created_at=time.time(),
            updated_at=time.time(),
        )

        assert asset.latest_version is None

    def test_latest_version_property_with_versions(self):
        """Test latest_version property with multiple versions."""
        asset = ManagedAsset(
            id="test_001",
            name="Test_Asset",
            original_prompt="Test prompt",
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Add versions in non-sequential order
        version1 = AssetVersion(version=1, timestamp=time.time(), file_path="/v1")
        version3 = AssetVersion(version=3, timestamp=time.time(), file_path="/v3")
        version2 = AssetVersion(version=2, timestamp=time.time(), file_path="/v2")

        asset.versions = [version1, version3, version2]

        latest = asset.latest_version
        assert latest is not None
        assert latest.version == 3

    def test_current_asset_version_property(self):
        """Test current_asset_version property."""
        asset = ManagedAsset(
            id="test_001",
            name="Test_Asset",
            original_prompt="Test prompt",
            created_at=time.time(),
            updated_at=time.time(),
            current_version=2,
        )

        version1 = AssetVersion(version=1, timestamp=time.time(), file_path="/v1")
        version2 = AssetVersion(version=2, timestamp=time.time(), file_path="/v2")
        version3 = AssetVersion(version=3, timestamp=time.time(), file_path="/v3")

        asset.versions = [version1, version2, version3]

        current = asset.current_asset_version
        assert current is not None
        assert current.version == 2

    def test_current_asset_version_not_found(self):
        """Test current_asset_version when version doesn't exist."""
        asset = ManagedAsset(
            id="test_001",
            name="Test_Asset",
            original_prompt="Test prompt",
            created_at=time.time(),
            updated_at=time.time(),
            current_version=5,  # Version that doesn't exist
        )

        version1 = AssetVersion(version=1, timestamp=time.time(), file_path="/v1")
        asset.versions = [version1]

        current = asset.current_asset_version
        assert current is None


class TestAssetRepository:
    """Test cases for AssetRepository."""

    def test_initialization(self, temp_repo_path):
        """Test repository initialization."""
        repo = AssetRepository(temp_repo_path)

        assert repo.base_path == Path(temp_repo_path)
        assert repo.assets_dir.exists()
        assert repo.metadata_dir.exists()
        assert repo.screenshots_dir.exists()
        assert isinstance(repo._assets_cache, dict)

    def test_directory_structure(self, temp_repo_path):
        """Test proper directory structure creation."""
        AssetRepository(temp_repo_path)

        expected_dirs = [
            Path(temp_repo_path) / "files",
            Path(temp_repo_path) / "metadata",
            Path(temp_repo_path) / "screenshots",
        ]

        for directory in expected_dirs:
            assert directory.exists()
            assert directory.is_dir()

    def test_generate_asset_name_normal(self, temp_repo_path):
        """Test asset name generation from normal prompt."""
        repo = AssetRepository(temp_repo_path)

        prompt = "create a red metallic cube with smooth edges"
        name = repo._generate_asset_name(prompt)

        assert name == "Create_A_Red_Metallic"

    def test_generate_asset_name_special_chars(self, temp_repo_path):
        """Test asset name generation with special characters."""
        repo = AssetRepository(temp_repo_path)

        prompt = "create a @#$% weird!! shape..."
        name = repo._generate_asset_name(prompt)

        assert name == "Create_A_Weird_Shape"

    def test_generate_asset_name_empty(self, temp_repo_path):
        """Test asset name generation with empty/invalid prompt."""
        repo = AssetRepository(temp_repo_path)

        prompt = "@#$%^&*()"  # No valid alphanumeric words
        name = repo._generate_asset_name(prompt)

        assert name == "UnnamedAsset"

    def test_create_asset_success(self, temp_repo_path, sample_asset_metadata):
        """Test successful asset creation."""
        repo = AssetRepository(temp_repo_path)

        managed_asset = repo.create_asset(
            sample_asset_metadata, quality_score=8.5, tags=["cube", "red"]
        )

        assert managed_asset.id == sample_asset_metadata.id
        assert managed_asset.name == "Create_A_Red_Cube"
        assert managed_asset.original_prompt == sample_asset_metadata.prompt
        assert len(managed_asset.subtasks) == 2
        assert managed_asset.tags == ["cube", "red"]
        assert len(managed_asset.versions) == 1
        assert managed_asset.current_version == 1

        # Check version details
        version = managed_asset.versions[0]
        assert version.version == 1
        assert version.quality_score == 8.5
        assert version.metadata["initial_creation"] is True

    def test_create_asset_file_storage(self, temp_repo_path, sample_asset_metadata):
        """Test asset file storage during creation."""
        repo = AssetRepository(temp_repo_path)

        managed_asset = repo.create_asset(sample_asset_metadata)
        version = managed_asset.versions[0]

        # Check that files were copied to repository
        stored_asset_file = Path(version.file_path)
        stored_screenshot_file = Path(version.screenshot_path)

        assert stored_asset_file.exists()
        assert stored_screenshot_file.exists()
        assert stored_asset_file.parent.name == managed_asset.id
        assert stored_screenshot_file.name.startswith(managed_asset.id)

    def test_create_asset_metadata_persistence(
        self, temp_repo_path, sample_asset_metadata
    ):
        """Test asset metadata persistence."""
        repo = AssetRepository(temp_repo_path)

        managed_asset = repo.create_asset(sample_asset_metadata)

        # Check metadata file exists
        metadata_file = repo.metadata_dir / f"{managed_asset.id}.json"
        assert metadata_file.exists()

        # Check metadata content
        with open(metadata_file) as f:
            saved_data = json.load(f)

        assert saved_data["id"] == managed_asset.id
        assert saved_data["name"] == managed_asset.name
        assert len(saved_data["subtasks"]) == 2
        assert len(saved_data["versions"]) == 1

    def test_add_version_success(self, temp_repo_path, sample_asset_metadata):
        """Test successful version addition."""
        repo = AssetRepository(temp_repo_path)

        # Create initial asset
        managed_asset = repo.create_asset(sample_asset_metadata)

        # Create new version metadata
        new_metadata = AssetMetadata(
            id=sample_asset_metadata.id,
            prompt="Refined version",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=sample_asset_metadata.subtasks,
        )

        # Add version
        new_version = repo.add_version(
            managed_asset.id,
            new_metadata,
            refinement_request="Make it shinier",
            quality_score=9.0,
        )

        assert new_version is not None
        assert new_version.version == 2
        assert new_version.refinement_request == "Make it shinier"
        assert new_version.quality_score == 9.0
        assert new_version.metadata["refinement"] is True

        # Check asset was updated
        updated_asset = repo.get_asset(managed_asset.id)
        assert len(updated_asset.versions) == 2
        assert updated_asset.current_version == 2

    def test_add_version_asset_not_found(self, temp_repo_path, sample_asset_metadata):
        """Test version addition with non-existent asset."""
        repo = AssetRepository(temp_repo_path)

        result = repo.add_version(
            "nonexistent_id", sample_asset_metadata, refinement_request="Test"
        )

        assert result is None

    def test_get_asset_success(self, temp_repo_path, sample_asset_metadata):
        """Test successful asset retrieval."""
        repo = AssetRepository(temp_repo_path)

        created_asset = repo.create_asset(sample_asset_metadata)
        retrieved_asset = repo.get_asset(created_asset.id)

        assert retrieved_asset is not None
        assert retrieved_asset.id == created_asset.id
        assert retrieved_asset.name == created_asset.name

    def test_get_asset_not_found(self, temp_repo_path):
        """Test asset retrieval with non-existent ID."""
        repo = AssetRepository(temp_repo_path)

        result = repo.get_asset("nonexistent_id")
        assert result is None

    def test_list_assets_no_filter(self, temp_repo_path, sample_asset_metadata):
        """Test listing all assets without filters."""
        repo = AssetRepository(temp_repo_path)

        # Create multiple assets
        asset1 = repo.create_asset(sample_asset_metadata, tags=["cube"])

        # Create second asset
        metadata2 = AssetMetadata(
            id="asset_002",
            prompt="Create a sphere",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )
        asset2 = repo.create_asset(metadata2, tags=["sphere"])

        assets = repo.list_assets()

        assert len(assets) == 2
        asset_ids = [asset.id for asset in assets]
        assert asset1.id in asset_ids
        assert asset2.id in asset_ids

    def test_list_assets_filter_by_tags(self, temp_repo_path, sample_asset_metadata):
        """Test listing assets filtered by tags."""
        repo = AssetRepository(temp_repo_path)

        # Create assets with different tags
        asset1 = repo.create_asset(sample_asset_metadata, tags=["cube", "red"])

        metadata2 = AssetMetadata(
            id="asset_002",
            prompt="Create a sphere",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )
        asset2 = repo.create_asset(metadata2, tags=["sphere", "blue"])

        # Filter by cube tag
        cube_assets = repo.list_assets(tags=["cube"])
        assert len(cube_assets) == 1
        assert cube_assets[0].id == asset1.id

        # Filter by blue tag
        blue_assets = repo.list_assets(tags=["blue"])
        assert len(blue_assets) == 1
        assert blue_assets[0].id == asset2.id

    def test_list_assets_filter_by_quality(self, temp_repo_path, sample_asset_metadata):
        """Test listing assets filtered by quality score."""
        repo = AssetRepository(temp_repo_path)

        # Create assets with different quality scores
        asset1 = repo.create_asset(sample_asset_metadata, quality_score=9.0)

        metadata2 = AssetMetadata(
            id="asset_002",
            prompt="Create a low quality sphere",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )
        repo.create_asset(metadata2, quality_score=6.0)

        # Filter by minimum quality
        high_quality_assets = repo.list_assets(min_quality_score=8.0)
        assert len(high_quality_assets) == 1
        assert high_quality_assets[0].id == asset1.id

    def test_list_assets_with_limit(self, temp_repo_path, sample_asset_metadata):
        """Test listing assets with limit."""
        repo = AssetRepository(temp_repo_path)

        # Create 5 assets
        for i in range(5):
            metadata = AssetMetadata(
                id=f"asset_{i:03d}",
                prompt=f"Create asset {i}",
                file_path=sample_asset_metadata.file_path,
                screenshot_path=sample_asset_metadata.screenshot_path,
                subtasks=[],
            )
            repo.create_asset(metadata)

        # List with limit
        limited_assets = repo.list_assets(limit=3)
        assert len(limited_assets) == 3

    def test_delete_asset_success(self, temp_repo_path, sample_asset_metadata):
        """Test successful asset deletion."""
        repo = AssetRepository(temp_repo_path)

        managed_asset = repo.create_asset(sample_asset_metadata)
        asset_id = managed_asset.id

        # Verify asset exists
        assert repo.get_asset(asset_id) is not None

        # Delete asset
        result = repo.delete_asset(asset_id)
        assert result is True

        # Verify asset is gone
        assert repo.get_asset(asset_id) is None

        # Verify metadata file is gone
        metadata_file = repo.metadata_dir / f"{asset_id}.json"
        assert not metadata_file.exists()

    def test_delete_asset_not_found(self, temp_repo_path):
        """Test asset deletion with non-existent ID."""
        repo = AssetRepository(temp_repo_path)

        result = repo.delete_asset("nonexistent_id")
        assert result is False

    def test_set_current_version_success(self, temp_repo_path, sample_asset_metadata):
        """Test successful current version change."""
        repo = AssetRepository(temp_repo_path)

        # Create asset with multiple versions
        managed_asset = repo.create_asset(sample_asset_metadata)
        repo.add_version(managed_asset.id, sample_asset_metadata)
        repo.add_version(managed_asset.id, sample_asset_metadata)

        # Set current version to 2
        result = repo.set_current_version(managed_asset.id, 2)
        assert result is True

        # Verify current version changed
        updated_asset = repo.get_asset(managed_asset.id)
        assert updated_asset.current_version == 2

    def test_set_current_version_not_found(self, temp_repo_path):
        """Test current version change with non-existent asset."""
        repo = AssetRepository(temp_repo_path)

        result = repo.set_current_version("nonexistent_id", 1)
        assert result is False

    def test_set_current_version_invalid_version(
        self, temp_repo_path, sample_asset_metadata
    ):
        """Test current version change with non-existent version."""
        repo = AssetRepository(temp_repo_path)

        managed_asset = repo.create_asset(sample_asset_metadata)

        # Try to set to version that doesn't exist
        result = repo.set_current_version(managed_asset.id, 99)
        assert result is False

    def test_get_asset_statistics(self, temp_repo_path, sample_asset_metadata):
        """Test asset statistics calculation."""
        repo = AssetRepository(temp_repo_path)

        # Create assets with different properties
        asset1 = repo.create_asset(
            sample_asset_metadata, quality_score=8.0, tags=["cube"]
        )

        metadata2 = AssetMetadata(
            id="asset_002",
            prompt="Create a sphere",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )
        repo.create_asset(metadata2, quality_score=9.0, tags=["sphere", "blue"])

        # Add version to first asset
        repo.add_version(asset1.id, sample_asset_metadata, quality_score=8.5)

        stats = repo.get_asset_statistics()

        assert stats["total_assets"] == 2
        assert stats["total_versions"] == 3  # 2 + 1 additional version
        assert stats["average_quality_score"] == 8.75  # (8.5 + 9.0) / 2
        assert stats["assets_with_quality_scores"] == 2
        assert stats["tag_counts"]["cube"] == 1
        assert stats["tag_counts"]["sphere"] == 1
        assert stats["tag_counts"]["blue"] == 1


class TestAssetManager:
    """Test cases for AssetManager."""

    def test_initialization(self, temp_repo_path):
        """Test asset manager initialization."""
        manager = AssetManager(temp_repo_path)

        assert hasattr(manager, "repository")
        assert isinstance(manager.repository, AssetRepository)

    def test_create_from_workflow_state(self, temp_repo_path, sample_asset_metadata):
        """Test asset creation from workflow state."""
        manager = AssetManager(temp_repo_path)

        # Mock workflow state
        workflow_state = Mock()
        workflow_state.asset_metadata = sample_asset_metadata
        workflow_state.verification_result = {"quality_score": 8.5}

        managed_asset = manager.create_from_workflow_state(
            workflow_state, tags=["test", "workflow"]
        )

        assert managed_asset is not None
        assert managed_asset.id == sample_asset_metadata.id
        assert "test" in managed_asset.tags
        assert "workflow" in managed_asset.tags

        # Check quality score was extracted
        version = managed_asset.latest_version
        assert version.quality_score == 8.5

    def test_create_from_workflow_state_no_metadata(self, temp_repo_path):
        """Test asset creation from workflow state without metadata."""
        manager = AssetManager(temp_repo_path)

        # Mock workflow state without asset_metadata
        workflow_state = Mock()
        del workflow_state.asset_metadata  # Simulate missing attribute

        result = manager.create_from_workflow_state(workflow_state)
        assert result is None

    def test_add_refinement_version(self, temp_repo_path, sample_asset_metadata):
        """Test adding refinement version."""
        manager = AssetManager(temp_repo_path)

        # Create initial asset
        workflow_state = Mock()
        workflow_state.asset_metadata = sample_asset_metadata
        workflow_state.verification_result = {"quality_score": 8.0}

        initial_asset = manager.create_from_workflow_state(workflow_state)

        # Add refinement version
        refined_metadata = AssetMetadata(
            id=sample_asset_metadata.id,
            prompt="Refined version",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=sample_asset_metadata.subtasks,
        )

        refined_state = Mock()
        refined_state.asset_metadata = refined_metadata
        refined_state.verification_result = {"quality_score": 9.0}

        new_version = manager.add_refinement_version(
            initial_asset.id, refined_state, "Make it more detailed"
        )

        assert new_version is not None
        assert new_version.version == 2
        assert new_version.refinement_request == "Make it more detailed"
        assert new_version.quality_score == 9.0

    def test_get_best_assets(self, temp_repo_path, sample_asset_metadata):
        """Test getting best quality assets."""
        manager = AssetManager(temp_repo_path)

        # Create assets with different quality scores
        high_quality_metadata = AssetMetadata(
            id="high_quality",
            prompt="High quality asset",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )

        low_quality_metadata = AssetMetadata(
            id="low_quality",
            prompt="Low quality asset",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )

        manager.repository.create_asset(high_quality_metadata, quality_score=9.5)
        manager.repository.create_asset(low_quality_metadata, quality_score=6.0)
        manager.repository.create_asset(sample_asset_metadata, quality_score=8.5)

        best_assets = manager.get_best_assets(limit=2)

        assert len(best_assets) == 2
        # Should be sorted by quality (but list_assets sorts by updated_at)
        # All assets should have quality >= 8.0
        for asset in best_assets:
            latest_version = asset.latest_version
            assert latest_version.quality_score >= 8.0

    def test_cleanup_low_quality_assets(self, temp_repo_path, sample_asset_metadata):
        """Test cleanup of low quality assets."""
        manager = AssetManager(temp_repo_path)

        # Create assets with different quality scores
        high_quality_metadata = AssetMetadata(
            id="high_quality",
            prompt="High quality asset",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )

        low_quality_metadata = AssetMetadata(
            id="low_quality",
            prompt="Low quality asset",
            file_path=sample_asset_metadata.file_path,
            screenshot_path=sample_asset_metadata.screenshot_path,
            subtasks=[],
        )

        manager.repository.create_asset(high_quality_metadata, quality_score=9.0)
        manager.repository.create_asset(low_quality_metadata, quality_score=3.0)
        manager.repository.create_asset(sample_asset_metadata, quality_score=8.0)

        # Cleanup assets with quality < 5.0
        deleted_count = manager.cleanup_low_quality_assets(min_quality=5.0)

        assert deleted_count == 1

        # Verify low quality asset was deleted
        remaining_assets = manager.repository.list_assets()
        assert len(remaining_assets) == 2

        remaining_ids = [asset.id for asset in remaining_assets]
        assert "low_quality" not in remaining_ids
        assert "high_quality" in remaining_ids
