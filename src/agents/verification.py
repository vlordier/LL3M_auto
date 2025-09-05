"""Verification agent for quality control and automated testing of 3D assets."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..blender.enhanced_executor import EnhancedBlenderExecutor
from ..blender.verification_scripts import (
    create_asset_analysis_script,
    create_benchmark_script,
    parse_analysis_results,
    parse_benchmark_results,
)
from ..utils.types import (
    AgentResponse,
    AgentType,
    IssueType,
    WorkflowState,
)
from .base import EnhancedBaseAgent
from .error_handling import AgentErrorHandler
from .utils import get_asset_path_safely

# Quality thresholds
MIN_POLYGON_COUNT = 100
MIN_QUALITY_POLYGONS = 500


@dataclass
class QualityMetrics:
    """Quality metrics for 3D asset verification."""

    geometry_valid: bool = False
    material_count: int = 0
    polygon_count: int = 0
    vertex_count: int = 0
    has_textures: bool = False
    has_lighting: bool = False
    scene_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_time: float = 0.0
    file_size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "geometry_valid": self.geometry_valid,
            "material_count": self.material_count,
            "polygon_count": self.polygon_count,
            "vertex_count": self.vertex_count,
            "has_textures": self.has_textures,
            "has_lighting": self.has_lighting,
            "scene_bounds": self.scene_bounds,
            "render_time": self.render_time,
            "file_size_mb": self.file_size_mb,
        }


@dataclass
class VerificationResult:
    """Result of 3D asset verification."""

    quality_score: float = 0.0
    issues_found: list[dict[str, Any]] | None = None
    metrics: QualityMetrics | None = None
    performance_benchmarks: dict[str, Any] | None = None
    recommendations: list[str] | None = None

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.issues_found is None:
            self.issues_found = []
        if self.metrics is None:
            self.metrics = QualityMetrics()
        if self.performance_benchmarks is None:
            self.performance_benchmarks = {}
        if self.recommendations is None:
            self.recommendations = []


class VerificationAgent(EnhancedBaseAgent):
    """Performs comprehensive quality control and automated testing of 3D assets."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize verification agent."""
        super().__init__(config)
        self.quality_thresholds = {
            "min_quality_score": 7.0,
            "max_polygon_count": 100000,
            "max_vertex_count": 150000,
            "max_file_size_mb": 50.0,
            "max_render_time": 10.0,
            "min_geometry_complexity": 0.1,
        }
        self.executor = EnhancedBlenderExecutor()

    @property
    def agent_type(self) -> AgentType:
        """Return agent type."""
        return AgentType.VERIFICATION

    @property
    def name(self) -> str:
        """Return agent name."""
        return "Quality Verification"

    async def process(self, state: WorkflowState) -> AgentResponse:
        """Verify 3D asset quality and performance."""
        start_time = asyncio.get_event_loop().time()

        try:
            if not await self.validate_input(state):
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=False,
                    data={},
                    message="Invalid input: execution result with asset file required",
                    execution_time=0.0,
                )

            self.logger.info("Starting asset verification")

            # Initialize verification result
            verification_result = VerificationResult()

            # Perform verification steps
            await self._analyze_asset_metrics(state, verification_result)
            await self._perform_quality_checks(state, verification_result)
            await self._run_performance_benchmarks(state, verification_result)
            await self._validate_against_requirements(state, verification_result)

            # Calculate overall quality score
            verification_result.quality_score = self._calculate_quality_score(verification_result)

            # Generate recommendations
            verification_result.recommendations = self._generate_recommendations(verification_result)

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(
                "Asset verification completed",
                quality_score=verification_result.quality_score,
                issues_count=len(verification_result.issues_found or []),
                execution_time=execution_time,
            )

            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data=verification_result.__dict__,
                message=self._generate_summary_message(verification_result),
                execution_time=execution_time,
                metadata={
                    "quality_score": verification_result.quality_score,
                    "issues_count": len(verification_result.issues_found or []),
                    "critical_issues": self._count_critical_issues(verification_result),
                },
            )

        except Exception as verification_error:
            AgentErrorHandler.log_error(
                self.logger,
                verification_error,
                "asset verification",
                has_execution_result=state.execution_result is not None,
            )
            return AgentErrorHandler.create_error_response(
                self.agent_type, verification_error, "asset verification", start_time
            )

    async def _analyze_asset_metrics(self, state: WorkflowState, result: VerificationResult) -> None:
        """Analyze basic asset metrics using Blender."""
        asset_path = get_asset_path_safely(state)
        if not asset_path or not Path(asset_path).exists():
            self._add_issue(result, IssueType.GEOMETRY_ERROR, "Asset file not found", "critical")
            return

        # Create Blender script for asset analysis
        analysis_script = create_asset_analysis_script()

        try:
            # Run Blender analysis using the executor
            execution_result = await self.executor.execute_code(
                code=analysis_script,
                asset_name=f"analysis_{Path(asset_path).stem}",
                validate_code=False,  # Internal script, no need to validate
            )

            if not execution_result.success:
                self._add_issue(
                    result,
                    IssueType.GEOMETRY_ERROR,
                    f"Blender analysis failed: {execution_result.errors}",
                    "high",
                )
                return

            # Parse metrics from output
            metrics_data = parse_analysis_results(execution_result.logs or [])
            if metrics_data:
                result.metrics = QualityMetrics(**metrics_data)
                result.metrics.file_size_mb = Path(asset_path).stat().st_size / (1024 * 1024)

        except Exception as asset_analysis_error:
            self._add_issue(
                result,
                IssueType.GEOMETRY_ERROR,
                f"Asset analysis error: {asset_analysis_error!s}",
                "high",
            )

    async def _perform_quality_checks(self, state: WorkflowState, result: VerificationResult) -> None:
        """Perform quality checks against thresholds."""
        metrics = result.metrics
        if not metrics:
            return

        # Check polygon count
        if metrics.polygon_count > self.quality_thresholds["max_polygon_count"]:
            self._add_issue(
                result,
                IssueType.GEOMETRY_ERROR,
                f"Polygon count too high: {metrics.polygon_count} > {self.quality_thresholds['max_polygon_count']}",
                "medium",
            )

        # Check vertex count
        if metrics.vertex_count > self.quality_thresholds["max_vertex_count"]:
            self._add_issue(
                result,
                IssueType.GEOMETRY_ERROR,
                f"Vertex count too high: {metrics.vertex_count} > {self.quality_thresholds['max_vertex_count']}",
                "medium",
            )

        # Check file size
        if metrics.file_size_mb > self.quality_thresholds["max_file_size_mb"]:
            self._add_issue(
                result,
                IssueType.SCALE_ISSUE,
                f"File size too large: {metrics.file_size_mb:.1f}MB > {self.quality_thresholds['max_file_size_mb']}MB",
                "low",
            )

        # Check geometry validity
        if not metrics.geometry_valid:
            self._add_issue(
                result,
                IssueType.GEOMETRY_ERROR,
                "Invalid geometry detected (non-manifold, degenerate faces)",
                "high",
            )

        # Check material setup
        if (
            state.subtasks
            and any(task.type.value == "material" for task in state.subtasks)
            and metrics.material_count == 0
        ):
            self._add_issue(
                result,
                IssueType.MATERIAL_ISSUE,
                "No materials found despite material tasks",
                "medium",
            )

        # Check lighting setup
        if (
            state.subtasks
            and any(task.type.value == "lighting" for task in state.subtasks)
            and not metrics.has_lighting
        ):
            self._add_issue(
                result,
                IssueType.LIGHTING_PROBLEM,
                "No lighting found despite lighting tasks",
                "medium",
            )

    async def _run_performance_benchmarks(self, state: WorkflowState, result: VerificationResult) -> None:
        """Run performance benchmarks on the asset."""
        asset_path = get_asset_path_safely(state)
        if not asset_path or not Path(asset_path).exists():
            return

        # Create benchmark script
        benchmark_script = create_benchmark_script(asset_path)

        try:
            # Run benchmark using the executor
            execution_result = await self.executor.execute_code(
                code=benchmark_script,
                asset_name=f"benchmark_{Path(asset_path).stem}",
                validate_code=False,  # Internal script
            )

            if execution_result.success:
                benchmark_data = parse_benchmark_results(execution_result.logs or [])
                result.performance_benchmarks = benchmark_data
                if result.performance_benchmarks is not None:
                    result.performance_benchmarks["total_benchmark_time"] = execution_result.execution_time

                # Check render time threshold
                render_time = benchmark_data.get("render_time", 0)
                if render_time > self.quality_thresholds["max_render_time"]:
                    self._add_issue(
                        result,
                        IssueType.SCALE_ISSUE,
                        f"Render time too slow: {render_time:.1f}s > {self.quality_thresholds['max_render_time']}s",
                        "low",
                    )
            else:
                self.logger.warning("Benchmark execution failed", errors=execution_result.errors)
                if result.performance_benchmarks is None:
                    result.performance_benchmarks = {}
                result.performance_benchmarks["error"] = str(execution_result.errors)

        except Exception as verification_exception:
            self.logger.warning("Benchmark execution failed", error=str(verification_exception))
            if result.performance_benchmarks is None:
                result.performance_benchmarks = {}
            result.performance_benchmarks["error"] = str(verification_exception)

    async def _validate_against_requirements(self, state: WorkflowState, result: VerificationResult) -> None:
        """Validate asset against original requirements."""
        if not state.subtasks:
            return

        # Check if all subtasks are represented in the asset
        expected_materials = len([task for task in state.subtasks if task.type.value == "material"])

        if expected_materials > 0 and result.metrics and result.metrics.material_count < expected_materials:
            self._add_issue(
                result,
                IssueType.MATERIAL_ISSUE,
                f"Expected {expected_materials} materials, found "
                f"{result.metrics.material_count if result.metrics else 0}",
                "medium",
            )

        # Check geometry requirements
        geometry_tasks = [task for task in state.subtasks if task.type.value == "geometry"]

        if geometry_tasks and result.metrics and result.metrics.polygon_count < MIN_POLYGON_COUNT:
            self._add_issue(
                result,
                IssueType.GEOMETRY_ERROR,
                "Geometry too simple for requirements",
                "medium",
            )

    def _calculate_quality_score(self, result: VerificationResult) -> float:
        """Calculate overall quality score based on metrics and issues."""
        base_score = 10.0

        # Deduct points for issues
        for issue in result.issues_found or []:
            severity = issue.get("severity", "low")
            if severity == "critical":
                base_score -= 3.0
            elif severity == "high":
                base_score -= 2.0
            elif severity == "medium":
                base_score -= 1.0
            else:  # low
                base_score -= 0.5

        # Bonus points for good metrics
        metrics = result.metrics
        if metrics and metrics.geometry_valid:
            base_score += 0.5
        if metrics and metrics.has_textures:
            base_score += 0.5
        if metrics and metrics.has_lighting:
            base_score += 0.5

        # Ensure score is within bounds
        return max(0.0, min(10.0, base_score))

    def _generate_recommendations(self, result: VerificationResult) -> list[str]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []

        # Analyze issues and provide specific recommendations
        for issue in result.issues_found or []:
            description = issue.get("description", "")

            if "polygon count too high" in description.lower():
                recommendations.append("Consider using mesh decimation or simplification techniques")
            elif "vertex count too high" in description.lower():
                recommendations.append("Optimize mesh topology by removing unnecessary vertices")
            elif "file size too large" in description.lower():
                recommendations.append("Compress textures or reduce mesh complexity to decrease file size")
            elif "no materials found" in description.lower():
                recommendations.append("Add proper materials and shading to improve visual quality")
            elif "no lighting found" in description.lower():
                recommendations.append("Add appropriate lighting setup for better scene illumination")
            elif "render time too slow" in description.lower():
                recommendations.append("Optimize materials and geometry for better rendering performance")

        # Add general quality improvements
        metrics = result.metrics
        if metrics and not metrics.has_textures and metrics.material_count > 0:
            recommendations.append("Add texture maps to materials for enhanced visual realism")

        if metrics and metrics.polygon_count < MIN_QUALITY_POLYGONS:
            recommendations.append("Consider adding more geometric detail for better visual quality")

        return list(set(recommendations))  # Remove duplicates

    def _add_issue(
        self,
        result: VerificationResult,
        issue_type: IssueType,
        description: str,
        severity: str,
    ) -> None:
        """Add an issue to the verification result."""
        if result.issues_found is not None:
            result.issues_found.append(
                {
                    "type": issue_type.value,
                    "description": description,
                    "severity": severity,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

    def _count_critical_issues(self, result: VerificationResult) -> int:
        """Count critical and high severity issues."""
        return len([issue for issue in (result.issues_found or []) if issue.get("severity") in ["critical", "high"]])

    def _generate_summary_message(self, result: VerificationResult) -> str:
        """Generate human-readable summary message."""
        quality_score = result.quality_score
        issues_count = len(result.issues_found or [])
        critical_count = self._count_critical_issues(result)

        summary = f"Quality score: {quality_score:.1f}/10"

        if issues_count > 0:
            summary += f" - {issues_count} issues found"
            if critical_count > 0:
                summary += f" ({critical_count} critical/high)"

        if quality_score >= self.quality_thresholds["min_quality_score"]:
            summary += " - Passed verification"
        else:
            summary += " - Failed verification"

        return summary

    async def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for verification agent."""
        if not state.execution_result:
            return False

        asset_path = get_asset_path_safely(state)
        if not asset_path:
            return False

        return Path(asset_path).exists()
