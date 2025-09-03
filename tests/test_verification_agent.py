"""Tests for VerificationAgent."""

import pytest

from src.agents.verification import (
    QualityMetrics,
    VerificationAgent,
    VerificationResult,
)
from src.utils.types import (
    AgentType,
    ExecutionResult,
    IssueType,
    SubTask,
    TaskType,
    WorkflowState,
)


@pytest.fixture
def verification_config():
    """Verification agent configuration fixture."""
    return {
        "blender_executable": "blender",
        "quality_thresholds": {
            "min_quality_score": 7.0,
            "max_polygon_count": 100000,
            "max_vertex_count": 150000,
            "max_file_size_mb": 50.0,
            "max_render_time": 10.0,
        },
    }


@pytest.fixture
def sample_blend_file(tmp_path):
    """Create a sample blend file."""
    blend_file = tmp_path / "test_asset.blend"
    # Create a minimal file
    blend_file.write_bytes(b"BLENDER test data")
    return str(blend_file)


@pytest.fixture
def workflow_state_with_asset(sample_blend_file, tmp_path):
    """WorkflowState with execution result and asset file."""
    execution_result = ExecutionResult(
        success=True,
        asset_path=sample_blend_file,
        screenshot_path=str(tmp_path / "screenshot.png"),
        logs=["Test log"],
        errors=[],
        execution_time=1.0,
    )

    subtasks = [
        SubTask(
            id="task1",
            type=TaskType.GEOMETRY,
            description="Create a cube",
            parameters={"shape": "cube"},
        ),
        SubTask(
            id="task2",
            type=TaskType.MATERIAL,
            description="Add red material",
            parameters={"color": [1.0, 0.0, 0.0]},
        ),
    ]

    state = WorkflowState(
        prompt="Create a red cube",
        execution_result=execution_result,
        subtasks=subtasks,
    )
    return state


class TestQualityMetrics:
    """Test cases for QualityMetrics."""

    def test_initialization_defaults(self):
        """Test QualityMetrics with default values."""
        metrics = QualityMetrics()

        assert metrics.geometry_valid is False
        assert metrics.material_count == 0
        assert metrics.polygon_count == 0
        assert metrics.vertex_count == 0
        assert metrics.has_textures is False
        assert metrics.has_lighting is False
        assert metrics.scene_bounds == (0.0, 0.0, 0.0)
        assert metrics.render_time == 0.0
        assert metrics.file_size_mb == 0.0

    def test_initialization_custom_values(self):
        """Test QualityMetrics with custom values."""
        metrics = QualityMetrics(
            geometry_valid=True,
            material_count=3,
            polygon_count=1000,
            vertex_count=500,
            has_textures=True,
            has_lighting=True,
            scene_bounds=(10.0, 5.0, 8.0),
            render_time=2.5,
            file_size_mb=15.2,
        )

        assert metrics.geometry_valid is True
        assert metrics.material_count == 3
        assert metrics.polygon_count == 1000
        assert metrics.vertex_count == 500
        assert metrics.has_textures is True
        assert metrics.has_lighting is True
        assert metrics.scene_bounds == (10.0, 5.0, 8.0)
        assert metrics.render_time == 2.5
        assert metrics.file_size_mb == 15.2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = QualityMetrics(
            geometry_valid=True, material_count=2, polygon_count=500
        )

        result = metrics.to_dict()

        expected = {
            "geometry_valid": True,
            "material_count": 2,
            "polygon_count": 500,
            "vertex_count": 0,
            "has_textures": False,
            "has_lighting": False,
            "scene_bounds": (0.0, 0.0, 0.0),
            "render_time": 0.0,
            "file_size_mb": 0.0,
        }

        assert result == expected


class TestVerificationResult:
    """Test cases for VerificationResult."""

    def test_initialization_defaults(self):
        """Test VerificationResult with default values."""
        result = VerificationResult()

        assert result.quality_score == 0.0
        assert result.issues_found == []
        assert isinstance(result.metrics, QualityMetrics)
        assert result.performance_benchmarks == {}
        assert result.recommendations == []

    def test_post_init_none_values(self):
        """Test __post_init__ with None values."""
        result = VerificationResult(
            issues_found=None,
            metrics=None,
            performance_benchmarks=None,
            recommendations=None,
        )

        assert result.issues_found == []
        assert isinstance(result.metrics, QualityMetrics)
        assert result.performance_benchmarks == {}
        assert result.recommendations == []


class TestVerificationAgent:
    """Test cases for VerificationAgent."""

    def test_initialization(self, verification_config):
        """Test agent initialization."""
        agent = VerificationAgent(verification_config)

        assert agent.agent_type == AgentType.VERIFICATION
        assert agent.name == "Quality Verification"
        assert hasattr(agent, "quality_thresholds")
        assert agent.blender_executable == "blender"

    def test_quality_thresholds(self, verification_config):
        """Test quality thresholds configuration."""
        agent = VerificationAgent(verification_config)

        expected = verification_config["quality_thresholds"]
        assert agent.quality_thresholds == expected

    @pytest.mark.asyncio
    async def test_validate_input_valid(
        self, verification_config, workflow_state_with_asset
    ):
        """Test input validation with valid state."""
        agent = VerificationAgent(verification_config)

        result = await agent.validate_input(workflow_state_with_asset)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_input_no_execution_result(self, verification_config):
        """Test input validation without execution result."""
        agent = VerificationAgent(verification_config)
        state = WorkflowState(prompt="test")

        result = await agent.validate_input(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_no_asset_path(self, verification_config, tmp_path):
        """Test input validation without asset path."""
        agent = VerificationAgent(verification_config)

        execution_result = ExecutionResult(
            success=True,
            asset_path=None,
            screenshot_path=str(tmp_path / "screenshot.png"),
            logs=[],
            errors=[],
            execution_time=1.0,
        )

        state = WorkflowState(prompt="test", execution_result=execution_result)

        result = await agent.validate_input(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_missing_asset_file(
        self, verification_config, tmp_path
    ):
        """Test input validation with missing asset file."""
        agent = VerificationAgent(verification_config)

        execution_result = ExecutionResult(
            success=True,
            asset_path="/nonexistent/asset.blend",
            screenshot_path=str(tmp_path / "screenshot.png"),
            logs=[],
            errors=[],
            execution_time=1.0,
        )

        state = WorkflowState(prompt="test", execution_result=execution_result)

        result = await agent.validate_input(state)
        assert result is False

    def test_calculate_quality_score_perfect(self, verification_config):
        """Test quality score calculation with no issues."""
        agent = VerificationAgent(verification_config)

        result = VerificationResult()
        result.metrics.geometry_valid = True
        result.metrics.has_textures = True
        result.metrics.has_lighting = True

        score = agent._calculate_quality_score(result)
        assert score == 10.0  # 10.0 base + 1.5 bonus, capped at 10.0

    def test_calculate_quality_score_with_issues(self, verification_config):
        """Test quality score calculation with issues."""
        agent = VerificationAgent(verification_config)

        result = VerificationResult()
        result.issues_found = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"},
        ]

        # Base 10.0 - 3.0 (critical) - 2.0 (high) - 1.0 (medium) - 0.5 (low) = 3.5
        score = agent._calculate_quality_score(result)
        assert score == 3.5

    def test_calculate_quality_score_minimum(self, verification_config):
        """Test quality score calculation doesn't go below 0."""
        agent = VerificationAgent(verification_config)

        result = VerificationResult()
        # Add many critical issues
        result.issues_found = [{"severity": "critical"} for _ in range(10)]

        score = agent._calculate_quality_score(result)
        assert score == 0.0

    def test_add_issue(self, verification_config):
        """Test adding an issue to verification result."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()

        agent._add_issue(
            result, IssueType.GEOMETRY_ERROR, "Test geometry issue", "high"
        )

        assert len(result.issues_found) == 1
        issue = result.issues_found[0]
        assert issue["type"] == "geometry_error"
        assert issue["description"] == "Test geometry issue"
        assert issue["severity"] == "high"
        assert "timestamp" in issue

    def test_count_critical_issues(self, verification_config):
        """Test counting critical and high severity issues."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()

        result.issues_found = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"},
            {"severity": "critical"},
        ]

        count = agent._count_critical_issues(result)
        assert count == 3  # 2 critical + 1 high

    def test_generate_summary_message_passed(self, verification_config):
        """Test summary message generation for passed verification."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()
        result.quality_score = 8.5
        result.issues_found = []

        message = agent._generate_summary_message(result)

        assert "Quality score: 8.5/10" in message
        assert "Passed verification" in message
        assert "issues found" not in message

    def test_generate_summary_message_failed_with_issues(self, verification_config):
        """Test summary message generation for failed verification with issues."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()
        result.quality_score = 5.0
        result.issues_found = [
            {"severity": "critical"},
            {"severity": "medium"},
            {"severity": "low"},
        ]

        message = agent._generate_summary_message(result)

        assert "Quality score: 5.0/10" in message
        assert "3 issues found" in message
        assert "(1 critical/high)" in message
        assert "Failed verification" in message

    def test_generate_recommendations_geometry_issues(self, verification_config):
        """Test recommendation generation for geometry issues."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()

        result.issues_found = [
            {
                "type": "geometry_error",
                "description": "Polygon count too high: 150000 > 100000",
                "severity": "medium",
            },
            {
                "type": "geometry_error",
                "description": "Vertex count too high: 200000 > 150000",
                "severity": "medium",
            },
        ]

        recommendations = agent._generate_recommendations(result)

        assert any("mesh decimation" in rec.lower() for rec in recommendations)
        assert any("unnecessary vertices" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_material_issues(self, verification_config):
        """Test recommendation generation for material issues."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()

        result.issues_found = [
            {
                "type": "material_issue",
                "description": "No materials found despite material tasks",
                "severity": "medium",
            }
        ]

        recommendations = agent._generate_recommendations(result)

        assert any(
            "materials" in rec.lower() and "shading" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_performance_issues(self, verification_config):
        """Test recommendation generation for performance issues."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()

        result.issues_found = [
            {
                "type": "scale_issue",
                "description": "Render time too slow: 15.0s > 10.0s",
                "severity": "low",
            },
            {
                "type": "scale_issue",
                "description": "File size too large: 60.0MB > 50.0MB",
                "severity": "low",
            },
        ]

        recommendations = agent._generate_recommendations(result)

        assert any(
            "optimize" in rec.lower() and "performance" in rec.lower()
            for rec in recommendations
        )
        assert any(
            "compress" in rec.lower() or "reduce" in rec.lower()
            for rec in recommendations
        )

    def test_generate_recommendations_quality_improvements(self, verification_config):
        """Test recommendation generation for quality improvements."""
        agent = VerificationAgent(verification_config)
        result = VerificationResult()

        # Low polygon count
        result.metrics.polygon_count = 200
        result.metrics.material_count = 2
        result.metrics.has_textures = False

        recommendations = agent._generate_recommendations(result)

        assert any("geometric detail" in rec.lower() for rec in recommendations)
        assert any("texture maps" in rec.lower() for rec in recommendations)

    def test_create_asset_analysis_script(self, verification_config):
        """Test asset analysis script creation."""
        agent = VerificationAgent(verification_config)

        script = agent._create_asset_analysis_script("/path/to/asset.blend")

        assert "import bpy" in script
        assert "import bmesh" in script
        assert "analyze_asset" in script
        assert "ANALYSIS_RESULTS" in script
        assert "material_count" in script
        assert "polygon_count" in script
        assert "vertex_count" in script

    def test_create_benchmark_script(self, verification_config):
        """Test benchmark script creation."""
        agent = VerificationAgent(verification_config)

        script = agent._create_benchmark_script("/path/to/asset.blend")

        assert "import bpy" in script
        assert "benchmark_asset" in script
        assert "BENCHMARK_RESULTS" in script
        assert "render_time" in script
        assert "bpy.ops.render.render" in script

    def test_parse_blender_output_success(self, verification_config):
        """Test parsing successful Blender analysis output."""
        agent = VerificationAgent(verification_config)

        output = """
        Some Blender output
        ANALYSIS_RESULTS: {"geometry_valid": true, "material_count": 2}
        More output
        """

        result = agent._parse_blender_output(output)

        assert result is not None
        assert result["geometry_valid"] is True
        assert result["material_count"] == 2

    def test_parse_blender_output_invalid_json(self, verification_config):
        """Test parsing Blender output with invalid JSON."""
        agent = VerificationAgent(verification_config)

        output = """
        Some Blender output
        ANALYSIS_RESULTS: {invalid json}
        More output
        """

        result = agent._parse_blender_output(output)

        assert result is None

    def test_parse_blender_output_no_marker(self, verification_config):
        """Test parsing Blender output without result marker."""
        agent = VerificationAgent(verification_config)

        output = "Just some regular Blender output"

        result = agent._parse_blender_output(output)

        assert result is None

    def test_parse_benchmark_output_success(self, verification_config):
        """Test parsing successful benchmark output."""
        agent = VerificationAgent(verification_config)

        output = """
        Benchmark output
        BENCHMARK_RESULTS: {"render_time": 2.5, "object_count": 5}
        """

        result = agent._parse_benchmark_output(output)

        assert result["render_time"] == 2.5
        assert result["object_count"] == 5

    def test_parse_benchmark_output_error(self, verification_config):
        """Test parsing benchmark output with error."""
        agent = VerificationAgent(verification_config)

        output = "No benchmark results found"

        result = agent._parse_benchmark_output(output)

        assert result == {}
