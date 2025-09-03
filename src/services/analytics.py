"""Advanced analytics and insights service."""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class GenerationMetrics(BaseModel):
    """Metrics for asset generation."""

    total_generations: int
    successful_generations: int
    failed_generations: int
    average_generation_time: float
    success_rate: float
    popular_prompts: list[str]
    generation_trends: dict[str, int]


class UserAnalytics(BaseModel):
    """User behavior analytics."""

    total_users: int
    active_users: int
    new_users_this_month: int
    user_retention_rate: float
    average_assets_per_user: float
    subscription_distribution: dict[str, int]


class SystemMetrics(BaseModel):
    """System performance metrics."""

    api_response_times: dict[str, float]
    blender_execution_times: list[float]
    queue_lengths: dict[str, int]
    error_rates: dict[str, float]
    resource_utilization: dict[str, float]


class QualityMetrics(BaseModel):
    """Asset quality metrics."""

    average_quality_score: float
    quality_improvement_rate: float
    refinement_success_rate: float
    common_issues: list[dict[str, Any]]
    user_satisfaction: float


class AnalyticsReport(BaseModel):
    """Comprehensive analytics report."""

    generation_metrics: GenerationMetrics
    user_analytics: UserAnalytics
    system_metrics: SystemMetrics
    quality_metrics: QualityMetrics
    report_generated_at: datetime


class AnalyticsService:
    """Service for generating analytics and insights."""

    def __init__(self):
        self.metrics_cache: dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=15)
        self.last_cache_update = datetime.utcnow()

    async def generate_analytics_report(
        self, time_range: timedelta | None = None, user_id: UUID | None = None
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report."""
        if time_range is None:
            time_range = timedelta(days=30)

        start_date = datetime.utcnow() - time_range
        end_date = datetime.utcnow()

        # Generate all metrics in parallel
        (
            generation_metrics,
            user_analytics,
            system_metrics,
            quality_metrics,
        ) = await asyncio.gather(
            self._get_generation_metrics(start_date, end_date, user_id),
            self._get_user_analytics(start_date, end_date)
            if user_id is None
            else self._get_empty_user_analytics(),
            self._get_system_metrics(start_date, end_date),
            self._get_quality_metrics(start_date, end_date, user_id),
        )

        return AnalyticsReport(
            generation_metrics=generation_metrics,
            user_analytics=user_analytics,
            system_metrics=system_metrics,
            quality_metrics=quality_metrics,
            report_generated_at=datetime.utcnow(),
        )

    async def _get_generation_metrics(
        self, start_date: datetime, end_date: datetime, user_id: UUID | None = None
    ) -> GenerationMetrics:
        """Get asset generation metrics."""
        cache_key = (
            f"generation_metrics_{start_date.date()}_{end_date.date()}_{user_id}"
        )

        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]

        # Mock data for demonstration (in production, query from database)
        total_generations = 1250
        successful_generations = 1187
        failed_generations = 63

        # Calculate metrics
        success_rate = (successful_generations / total_generations) * 100
        average_generation_time = 185.5  # seconds

        popular_prompts = [
            "futuristic robot with glowing eyes",
            "medieval castle on a mountain",
            "modern chair design",
            "abstract sculpture",
            "fantasy sword with runes",
        ]

        # Generation trends by day
        generation_trends = {}
        for i in range(7):
            date = end_date - timedelta(days=i)
            generation_trends[date.strftime("%Y-%m-%d")] = max(
                50 + i * 10, 180 - i * 15
            )

        metrics = GenerationMetrics(
            total_generations=total_generations,
            successful_generations=successful_generations,
            failed_generations=failed_generations,
            average_generation_time=average_generation_time,
            success_rate=success_rate,
            popular_prompts=popular_prompts,
            generation_trends=generation_trends,
        )

        self.metrics_cache[cache_key] = metrics
        return metrics

    async def _get_user_analytics(
        self, start_date: datetime, end_date: datetime
    ) -> UserAnalytics:
        """Get user behavior analytics."""
        cache_key = f"user_analytics_{start_date.date()}_{end_date.date()}"

        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]

        # Mock data for demonstration
        analytics = UserAnalytics(
            total_users=3247,
            active_users=892,
            new_users_this_month=234,
            user_retention_rate=67.8,
            average_assets_per_user=4.2,
            subscription_distribution={"free": 2456, "pro": 651, "enterprise": 140},
        )

        self.metrics_cache[cache_key] = analytics
        return analytics

    async def _get_system_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> SystemMetrics:
        """Get system performance metrics."""
        cache_key = f"system_metrics_{start_date.date()}_{end_date.date()}"

        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]

        # Mock data for demonstration
        metrics = SystemMetrics(
            api_response_times={
                "GET /api/v1/assets": 0.145,
                "POST /api/v1/assets/generate": 0.892,
                "GET /api/v1/health": 0.023,
                "POST /api/v1/auth/login": 0.234,
            },
            blender_execution_times=[120.5, 145.2, 189.7, 156.8, 142.3, 167.9, 198.1],
            queue_lengths={
                "generation_queue": 12,
                "refinement_queue": 5,
                "export_queue": 3,
            },
            error_rates={
                "api_errors": 0.8,  # percentage
                "generation_failures": 5.2,
                "timeout_errors": 1.1,
            },
            resource_utilization={
                "cpu_usage": 68.5,
                "memory_usage": 72.3,
                "disk_usage": 45.7,
                "gpu_usage": 82.1,
            },
        )

        self.metrics_cache[cache_key] = metrics
        return metrics

    async def _get_quality_metrics(
        self, start_date: datetime, end_date: datetime, user_id: UUID | None = None
    ) -> QualityMetrics:
        """Get asset quality metrics."""
        cache_key = f"quality_metrics_{start_date.date()}_{end_date.date()}_{user_id}"

        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]

        # Mock data for demonstration
        metrics = QualityMetrics(
            average_quality_score=7.8,  # out of 10
            quality_improvement_rate=12.5,  # percentage improvement
            refinement_success_rate=89.3,
            common_issues=[
                {"issue": "Poor lighting", "frequency": 23, "severity": "medium"},
                {"issue": "Low polygon count", "frequency": 18, "severity": "low"},
                {"issue": "Material issues", "frequency": 15, "severity": "high"},
                {"issue": "Scale problems", "frequency": 12, "severity": "medium"},
                {"issue": "Topology errors", "frequency": 8, "severity": "high"},
            ],
            user_satisfaction=8.2,  # out of 10
        )

        self.metrics_cache[cache_key] = metrics
        return metrics

    async def _get_empty_user_analytics(self) -> UserAnalytics:
        """Return empty user analytics for individual user reports."""
        return UserAnalytics(
            total_users=0,
            active_users=0,
            new_users_this_month=0,
            user_retention_rate=0.0,
            average_assets_per_user=0.0,
            subscription_distribution={},
        )

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.metrics_cache:
            return False

        time_since_update = datetime.utcnow() - self.last_cache_update
        return time_since_update < self.cache_ttl

    async def get_real_time_metrics(self) -> dict[str, Any]:
        """Get real-time system metrics."""
        import psutil

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Mock application metrics (would come from monitoring system)
        active_connections = 45
        requests_per_minute = 127
        active_generations = 8
        queue_depth = 15

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": (disk.used / disk.total) * 100,
                "available_memory": memory.available,
            },
            "application": {
                "active_connections": active_connections,
                "requests_per_minute": requests_per_minute,
                "active_generations": active_generations,
                "queue_depth": queue_depth,
                "uptime": 3600 * 24 * 5,  # 5 days in seconds
            },
        }

    async def analyze_generation_patterns(
        self, user_id: UUID, time_range: timedelta = timedelta(days=30)
    ) -> dict[str, Any]:
        """Analyze user's generation patterns and provide insights."""
        start_date = datetime.utcnow() - time_range

        # Mock user generation data
        user_generations = [
            {"prompt": "robot", "success": True, "time": 145.2, "quality": 8.1},
            {"prompt": "castle", "success": True, "time": 189.7, "quality": 7.9},
            {"prompt": "chair", "success": False, "time": None, "quality": None},
            {"prompt": "sculpture", "success": True, "time": 167.3, "quality": 8.5},
        ]

        # Analyze patterns
        successful_generations = [g for g in user_generations if g["success"]]
        failed_generations = [g for g in user_generations if not g["success"]]

        avg_generation_time = sum(g["time"] for g in successful_generations) / len(
            successful_generations
        )
        avg_quality = sum(g["quality"] for g in successful_generations) / len(
            successful_generations
        )

        # Generate insights
        insights = []

        if avg_generation_time > 180:
            insights.append(
                {
                    "type": "performance",
                    "message": "Your generations take longer than average. Try simpler prompts for faster results.",
                    "severity": "info",
                }
            )

        if avg_quality < 8.0:
            insights.append(
                {
                    "type": "quality",
                    "message": "Try adding more descriptive details to your prompts for better quality.",
                    "severity": "tip",
                }
            )

        if len(failed_generations) > len(successful_generations) * 0.2:
            insights.append(
                {
                    "type": "success_rate",
                    "message": "Consider breaking complex prompts into simpler parts.",
                    "severity": "warning",
                }
            )

        return {
            "user_id": str(user_id),
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": datetime.utcnow().isoformat(),
            },
            "statistics": {
                "total_generations": len(user_generations),
                "successful_generations": len(successful_generations),
                "success_rate": (len(successful_generations) / len(user_generations))
                * 100,
                "average_generation_time": avg_generation_time,
                "average_quality_score": avg_quality,
            },
            "insights": insights,
            "recommendations": [
                "Try using more specific adjectives in your prompts",
                "Experiment with different art styles",
                "Break complex objects into multiple simpler parts",
            ],
        }

    async def clear_cache(self):
        """Clear analytics cache."""
        self.metrics_cache.clear()
        self.last_cache_update = datetime.utcnow()


# Global analytics service instance
analytics_service = AnalyticsService()
