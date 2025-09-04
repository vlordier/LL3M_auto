"""Tests for asset management system."""

import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from pydantic import ValidationError

from src.assets.manager import (
    AssetManager,
    AssetRepository,
    AssetVersion,
    ManagedAsset,
)
from src.utils.types import AssetMetadata, SubTask, TaskType


@pytest.fixture
def sample_asset_metadata():
    """Return sample asset metadata for testing."""
    return AssetMetadata(
        id="test-asset-123",
        prompt="Create a simple cube",
        file_path="/tmp/test_asset.blend",
        screenshot_path="/tmp/test_screenshot.png",
        subtasks=[
            SubTask(
                id="task-1",
                type=TaskType.GEOMETRY,
                description="Create cube geometry",
                priority=1,
                parameters={"shape": "cube"},
            )
        ],
    )


@pytest.fixture
def tmp_repository(tmp_path):
    """Create a temporary asset repository for testing."""
    return AssetRepository(str(tmp_path / "test_assets"))


class TestAssetVersion:
    """Test AssetVersion data model."""

    def test_asset_version_creation(self):
        """Test creating an asset version."""
        version = AssetVersion(
            version=1,
            timestamp=1234567890.0,
            file_path="/path/to/asset.blend",
            screenshot_path="/path/to/screenshot.png",
            refinement_request="Improve lighting",
            quality_score=8.5,
        )

        assert version.version == 1
        assert version.timestamp == 1234567890.0
        assert version.file_path == "/path/to/asset.blend"
        assert version.screenshot_path == "/path/to/screenshot.png"
        assert version.refinement_request == "Improve lighting"
        assert version.quality_score == 8.5

    def test_asset_version_validation(self):
        """Test asset version validation."""
        # Test invalid version number
        with pytest.raises(ValidationError):
            AssetVersion(
                version=0,  # Should be >= 1
                timestamp=1234567890.0,
                file_path="/path/to/asset.blend",
            )

        # Test invalid quality score
        with pytest.raises(ValidationError):
            AssetVersion(
                version=1,
                timestamp=1234567890.0,
                file_path="/path/to/asset.blend",
                quality_score=11.0,  # Should be <= 10
            )


class TestManagedAsset:
    """Test ManagedAsset data model."""

    def test_managed_asset_creation(self, sample_asset_metadata):
        """Test creating a managed asset."""
        asset = ManagedAsset(
            id="test-123",
            name="Test_Cube",
            original_prompt="Create a cube",
            created_at=1234567890.0,
            updated_at=1234567890.0,
            subtasks=sample_asset_metadata.subtasks,
        )

        assert asset.id == "test-123"
        assert asset.name == "Test_Cube"
        assert asset.original_prompt == "Create a cube"
        assert len(asset.subtasks) == 1
        assert asset.current_version == 1

    def test_latest_version_empty(self):
        """Test getting latest version when no versions exist."""
        asset = ManagedAsset(
            id="test",
            name="Test",
            original_prompt="test",
            created_at=time.time(),
            updated_at=time.time(),
        )

        assert asset.latest_version is None

    def test_latest_version_with_versions(self):
        """Test getting latest version with multiple versions."""
        asset = ManagedAsset(
            id="test",
            name="Test",
            original_prompt="test",
            created_at=time.time(),
            updated_at=time.time(),
            versions=[
                AssetVersion(version=1, timestamp=1.0, file_path="v1.blend"),
                AssetVersion(version=3, timestamp=3.0, file_path="v3.blend"),
                AssetVersion(version=2, timestamp=2.0, file_path="v2.blend"),
            ],
        )

        latest = asset.latest_version
        assert latest is not None
        assert latest.version == 3

    def test_current_asset_version(self):
        """Test getting current active version."""
        asset = ManagedAsset(
            id="test",
            name="Test",
            original_prompt="test",
            created_at=time.time(),
            updated_at=time.time(),
            current_version=2,
            versions=[
                AssetVersion(version=1, timestamp=1.0, file_path="v1.blend"),
                AssetVersion(version=2, timestamp=2.0, file_path="v2.blend"),
                AssetVersion(version=3, timestamp=3.0, file_path="v3.blend"),
            ],
        )

        current = asset.current_asset_version
        assert current is not None
        assert current.version == 2

    def test_current_asset_version_not_found(self):
        """Test getting current version when it doesn't exist."""
        asset = ManagedAsset(
            id="test",
            name="Test",
            original_prompt="test",
            created_at=time.time(),
            updated_at=time.time(),
            current_version=5,  # Version doesn't exist
            versions=[AssetVersion(version=1, timestamp=1.0, file_path="v1.blend")],
        )

        assert asset.current_asset_version is None


class TestAssetRepository:
    """Test AssetRepository class."""

    def test_repository_initialization(self, tmp_path):
        """Test repository initialization creates directories."""
        repo_path = tmp_path / "test_repo"
        repo = AssetRepository(str(repo_path))

        assert repo.base_path.exists()
        assert repo.assets_dir.exists()
        assert repo.metadata_dir.exists()
        assert repo.screenshots_dir.exists()

    def test_generate_asset_name(self, tmp_repository):
        """Test asset name generation from prompt."""
        # Test normal prompt
        name = tmp_repository._generate_asset_name("Create a beautiful red car")
        assert name == "Create_A_Beautiful_Red"

        # Test prompt with special characters
        name = tmp_repository._generate_asset_name("Create! a @#$ cube++")
        assert name == "Create_A_Cube"

        # Test empty/invalid prompt
        name = tmp_repository._generate_asset_name("!@#$%")
        assert name == "UnnamedAsset"

    def test_create_asset(self, tmp_repository, sample_asset_metadata):
        """Test creating a new asset."""
        # Create temporary files
        Path("/tmp/test_asset.blend")
        Path("/tmp/test_screenshot.png")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.copy2"),
        ):
            asset = tmp_repository.create_asset(
                sample_asset_metadata, quality_score=8.5, tags=["cube", "geometry"]
            )

            assert asset.id == "test-asset-123"
            assert asset.name == "Create_A_Simple_Cube"
            assert asset.original_prompt == "Create a simple cube"
            assert len(asset.versions) == 1
            assert asset.versions[0].quality_score == 8.5
            assert asset.tags == ["cube", "geometry"]

    def test_create_asset_minimal(self, tmp_repository, sample_asset_metadata):
        """Test creating asset with minimal parameters."""
        sample_asset_metadata.file_path = None
        sample_asset_metadata.screenshot_path = None

        asset = tmp_repository.create_asset(sample_asset_metadata)

        assert asset.id == "test-asset-123"
        assert len(asset.versions) == 1
        assert asset.versions[0].file_path == ""
        assert asset.versions[0].screenshot_path is None

    def test_add_version(self, tmp_repository, sample_asset_metadata):
        """Test adding a new version to existing asset."""
        # First create an asset
        with patch("pathlib.Path.exists", return_value=True), patch("shutil.copy2"):
            original_asset = tmp_repository.create_asset(sample_asset_metadata)
            assert len(original_asset.versions) == 1

            # Add a new version
            new_metadata = AssetMetadata(
                id="test-asset-123",
                prompt="Refined cube",
                file_path="/tmp/refined_asset.blend",
            )

            new_version = tmp_repository.add_version(
                "test-asset-123",
                new_metadata,
                refinement_request="Make it bigger",
                quality_score=9.0,
            )

            assert new_version is not None
            assert new_version.version == 2
            assert new_version.refinement_request == "Make it bigger"
            assert new_version.quality_score == 9.0

            # Check asset was updated
            updated_asset = tmp_repository.get_asset("test-asset-123")
            assert len(updated_asset.versions) == 2
            assert updated_asset.current_version == 2

    def test_add_version_nonexistent_asset(self, tmp_repository, sample_asset_metadata):
        """Test adding version to non-existent asset."""
        result = tmp_repository.add_version(
            "nonexistent-id", sample_asset_metadata, refinement_request="test"
        )

        assert result is None

    def test_get_asset(self, tmp_repository, sample_asset_metadata):
        """Test getting asset by ID."""
        with patch("pathlib.Path.exists", return_value=True), patch("shutil.copy2"):
            created_asset = tmp_repository.create_asset(sample_asset_metadata)

            retrieved_asset = tmp_repository.get_asset("test-asset-123")
            assert retrieved_asset is not None
            assert retrieved_asset.id == created_asset.id

            # Test non-existent asset
            assert tmp_repository.get_asset("nonexistent") is None

    def test_list_assets_empty(self, tmp_repository):
        """Test listing assets when repository is empty."""
        assets = tmp_repository.list_assets()
        assert assets == []

    def test_list_assets_with_filters(self, tmp_repository, sample_asset_metadata):
        """Test listing assets with various filters."""
        with patch("pathlib.Path.exists", return_value=True), patch("shutil.copy2"):
            # Create multiple assets
            tmp_repository.create_asset(
                sample_asset_metadata, quality_score=8.0, tags=["cube", "geometry"]
            )

            sample_asset_metadata.id = "test-asset-456"
            tmp_repository.create_asset(
                sample_asset_metadata, quality_score=6.0, tags=["sphere", "geometry"]
            )

            sample_asset_metadata.id = "test-asset-789"
            tmp_repository.create_asset(
                sample_asset_metadata, quality_score=9.0, tags=["cube", "advanced"]
            )

            # Test no filters
            all_assets = tmp_repository.list_assets()
            assert len(all_assets) == 3

            # Test tag filter
            cube_assets = tmp_repository.list_assets(tags=["cube"])
            assert len(cube_assets) == 2

            # Test quality filter
            high_quality = tmp_repository.list_assets(min_quality_score=7.5)
            assert len(high_quality) == 2

            # Test limit
            limited = tmp_repository.list_assets(limit=2)
            assert len(limited) == 2

    def test_delete_asset(self, tmp_repository, sample_asset_metadata):
        """Test deleting an asset."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.copy2"),
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            # Create asset
            tmp_repository.create_asset(sample_asset_metadata)
            assert tmp_repository.get_asset("test-asset-123") is not None

            # Delete asset
            result = tmp_repository.delete_asset("test-asset-123")
            assert result is True

            # Verify deletion
            assert tmp_repository.get_asset("test-asset-123") is None
            mock_unlink.assert_called()

            # Test deleting non-existent asset
            result = tmp_repository.delete_asset("nonexistent")
            assert result is False

    def test_delete_version(self, tmp_repository, sample_asset_metadata):
        """Test deleting a specific version."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.copy2"),
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            # Create asset with multiple versions
            asset = tmp_repository.create_asset(sample_asset_metadata)

            new_metadata = AssetMetadata(
                id="test-asset-123", prompt="v2", file_path="/tmp/v2.blend"
            )
            tmp_repository.add_version("test-asset-123", new_metadata)

            asset = tmp_repository.get_asset("test-asset-123")
            assert len(asset.versions) == 2

            # Delete version 1
            result = tmp_repository.delete_version("test-asset-123", 1)
            assert result is True

            # Verify version deletion
            asset = tmp_repository.get_asset("test-asset-123")
            assert len(asset.versions) == 1
            assert asset.current_version == 2

            mock_unlink.assert_called()

    def test_delete_last_version(self, tmp_repository, sample_asset_metadata):
        """Test deleting the last version removes entire asset."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.copy2"),
            patch("pathlib.Path.unlink"),
        ):
            tmp_repository.create_asset(sample_asset_metadata)

            # Delete the only version
            result = tmp_repository.delete_version("test-asset-123", 1)
            assert result is True

            # Asset should be completely removed
            assert tmp_repository.get_asset("test-asset-123") is None

    def test_set_current_version(self, tmp_repository, sample_asset_metadata):
        """Test setting current active version."""
        with patch("pathlib.Path.exists", return_value=True), patch("shutil.copy2"):
            # Create asset with multiple versions
            asset = tmp_repository.create_asset(sample_asset_metadata)

            new_metadata = AssetMetadata(
                id="test-asset-123", prompt="v2", file_path="/tmp/v2.blend"
            )
            tmp_repository.add_version("test-asset-123", new_metadata)

            new_metadata = AssetMetadata(
                id="test-asset-123", prompt="v3", file_path="/tmp/v3.blend"
            )
            tmp_repository.add_version("test-asset-123", new_metadata)

            # Set current version to 1
            result = tmp_repository.set_current_version("test-asset-123", 1)
            assert result is True

            asset = tmp_repository.get_asset("test-asset-123")
            assert asset.current_version == 1

            # Test invalid version
            result = tmp_repository.set_current_version("test-asset-123", 99)
            assert result is False

            # Test non-existent asset
            result = tmp_repository.set_current_version("nonexistent", 1)
            assert result is False

    def test_get_asset_statistics(self, tmp_repository, sample_asset_metadata):
        """Test getting repository statistics."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.copy2"),
            patch.object(tmp_repository, "_calculate_total_size", return_value=10.5),
        ):
            # Empty repository
            stats = tmp_repository.get_asset_statistics()
            assert stats["total_assets"] == 0
            assert stats["total_versions"] == 0

            # Add some assets
            tmp_repository.create_asset(
                sample_asset_metadata, quality_score=8.0, tags=["cube", "geometry"]
            )

            sample_asset_metadata.id = "asset-2"
            tmp_repository.create_asset(
                sample_asset_metadata, quality_score=9.0, tags=["sphere"]
            )

            stats = tmp_repository.get_asset_statistics()
            assert stats["total_assets"] == 2
            assert stats["total_versions"] == 2
            assert stats["total_size_mb"] == 10.5
            assert stats["average_quality_score"] == 8.5
            assert stats["assets_with_quality_scores"] == 2
            assert stats["tag_counts"]["cube"] == 1
            assert stats["tag_counts"]["geometry"] == 1
            assert stats["tag_counts"]["sphere"] == 1

    def test_calculate_total_size(self, tmp_repository):
        """Test calculating total repository size."""
        # For simplicity, just test that it doesn't crash and returns a float
        size_mb = tmp_repository._calculate_total_size()
        assert isinstance(size_mb, float)
        assert size_mb >= 0.0

    def test_calculate_total_size_error(self, tmp_repository):
        """Test calculate total size with error handling."""
        with patch("pathlib.Path.iterdir", side_effect=Exception("Test error")):
            size_mb = tmp_repository._calculate_total_size()
            assert size_mb == 0.0

    def test_store_asset_files(self, tmp_repository, sample_asset_metadata):
        """Test storing asset files in repository."""
        asset = ManagedAsset(
            id="test-asset",
            name="Test",
            original_prompt="test",
            created_at=time.time(),
            updated_at=time.time(),
        )

        version = AssetVersion(version=1, timestamp=time.time(), file_path="")

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.copy2") as mock_copy,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            tmp_repository._store_asset_files(asset, version, sample_asset_metadata)

            # Should create asset directory and copy files
            mock_mkdir.assert_called_once()
            assert mock_copy.call_count == 2  # Asset file + screenshot

    def test_delete_version_files(self, tmp_repository):
        """Test deleting version files."""
        version = AssetVersion(
            version=1,
            timestamp=time.time(),
            file_path="/path/to/asset.blend",
            screenshot_path="/path/to/screenshot.png",
        )

        with patch("pathlib.Path.unlink") as mock_unlink:
            tmp_repository._delete_version_files(version)

            assert mock_unlink.call_count == 2

    def test_save_asset_metadata(self, tmp_repository):
        """Test saving asset metadata to disk."""
        asset = ManagedAsset(
            id="test-asset",
            name="Test",
            original_prompt="test",
            created_at=time.time(),
            updated_at=time.time(),
        )

        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
        ):
            tmp_repository._save_asset_metadata(asset)

            mock_file.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_load_assets_index(self, tmp_repository):
        """Test loading assets index from metadata files."""
        mock_asset_data = {
            "id": "test-asset",
            "name": "Test",
            "original_prompt": "test",
            "created_at": time.time(),
            "updated_at": time.time(),
            "versions": [],
            "tags": [],
            "subtasks": [],
            "current_version": 1,
        }

        with (
            patch("pathlib.Path.glob") as mock_glob,
            patch("builtins.open", mock_open()),
            patch("json.load", return_value=mock_asset_data),
        ):
            mock_file = MagicMock()
            mock_file.__str__ = lambda _: "test.json"
            mock_glob.return_value = [mock_file]

            tmp_repository._load_assets_index()

            assert "test-asset" in tmp_repository._assets_cache

    def test_load_assets_index_error(self, tmp_repository):
        """Test loading assets index with corrupted metadata."""
        with (
            patch("pathlib.Path.glob") as mock_glob,
            patch("builtins.open", mock_open()),
            patch("json.load", side_effect=Exception("Corrupted JSON")),
        ):
            mock_file = MagicMock()
            mock_file.__str__ = lambda _: "corrupted.json"
            mock_glob.return_value = [mock_file]

            # Should handle error gracefully
            tmp_repository._load_assets_index()

            assert len(tmp_repository._assets_cache) == 0


class TestAssetManager:
    """Test AssetManager high-level interface."""

    @pytest.fixture
    def asset_manager(self, tmp_path):
        """Create asset manager with temporary repository."""
        return AssetManager(str(tmp_path / "test_assets"))

    def test_asset_manager_initialization(self, asset_manager):
        """Test asset manager initialization."""
        assert asset_manager.repository is not None

    def test_create_from_workflow_state(self, asset_manager):
        """Test creating asset from workflow state."""
        # Mock workflow state
        workflow_state = MagicMock()
        workflow_state.asset_metadata = AssetMetadata(
            id="workflow-asset",
            prompt="Create from workflow",
            file_path="/tmp/workflow.blend",
        )
        workflow_state.verification_result = {"quality_score": 8.5}

        with patch.object(asset_manager.repository, "create_asset") as mock_create:
            mock_create.return_value = MagicMock()

            result = asset_manager.create_from_workflow_state(
                workflow_state, tags=["workflow", "test"]
            )

            mock_create.assert_called_once_with(
                workflow_state.asset_metadata,
                quality_score=8.5,
                tags=["workflow", "test"],
            )
            assert result is not None

    def test_create_from_workflow_state_no_metadata(self, asset_manager):
        """Test creating from workflow state without metadata."""
        workflow_state = MagicMock()
        del workflow_state.asset_metadata

        result = asset_manager.create_from_workflow_state(workflow_state)
        assert result is None

    def test_create_from_workflow_state_no_verification(self, asset_manager):
        """Test creating from workflow state without verification result."""
        workflow_state = MagicMock()
        workflow_state.asset_metadata = AssetMetadata(
            id="workflow-asset",
            prompt="Create from workflow",
            file_path="/tmp/workflow.blend",
        )
        del workflow_state.verification_result

        with patch.object(asset_manager.repository, "create_asset") as mock_create:
            mock_create.return_value = MagicMock()

            asset_manager.create_from_workflow_state(workflow_state)

            mock_create.assert_called_once_with(
                workflow_state.asset_metadata, quality_score=None, tags=None
            )

    def test_add_refinement_version(self, asset_manager):
        """Test adding refinement version."""
        workflow_state = MagicMock()
        workflow_state.asset_metadata = AssetMetadata(
            id="refined-asset", prompt="Refined version", file_path="/tmp/refined.blend"
        )
        workflow_state.verification_result = {"quality_score": 9.0}

        with patch.object(asset_manager.repository, "add_version") as mock_add:
            mock_add.return_value = MagicMock()

            result = asset_manager.add_refinement_version(
                "asset-123", workflow_state, "Make it better"
            )

            mock_add.assert_called_once_with(
                "asset-123",
                workflow_state.asset_metadata,
                refinement_request="Make it better",
                quality_score=9.0,
            )
            assert result is not None

    def test_get_best_assets(self, asset_manager):
        """Test getting best quality assets."""
        mock_assets = [MagicMock(), MagicMock()]

        with patch.object(asset_manager.repository, "list_assets") as mock_list:
            mock_list.return_value = mock_assets

            result = asset_manager.get_best_assets(limit=5)

            mock_list.assert_called_once_with(min_quality_score=8.0, limit=5)
            assert result == mock_assets

    def test_cleanup_low_quality_assets(self, asset_manager):
        """Test cleaning up low quality assets."""
        # Mock assets with different quality scores
        low_quality_asset = MagicMock()
        low_quality_asset.id = "low-quality"
        low_quality_version = MagicMock()
        low_quality_version.quality_score = 3.0
        low_quality_asset.latest_version = low_quality_version

        high_quality_asset = MagicMock()
        high_quality_asset.id = "high-quality"
        high_quality_version = MagicMock()
        high_quality_version.quality_score = 8.0
        high_quality_asset.latest_version = high_quality_version

        no_score_asset = MagicMock()
        no_score_asset.id = "no-score"
        no_score_version = MagicMock()
        no_score_version.quality_score = None
        no_score_asset.latest_version = no_score_version

        mock_assets = [low_quality_asset, high_quality_asset, no_score_asset]

        with (
            patch.object(asset_manager.repository, "list_assets") as mock_list,
            patch.object(asset_manager.repository, "delete_asset") as mock_delete,
        ):
            mock_list.return_value = mock_assets
            mock_delete.return_value = True

            deleted_count = asset_manager.cleanup_low_quality_assets(min_quality=5.0)

            assert deleted_count == 1
            mock_delete.assert_called_once_with("low-quality")
