"""Performance monitoring and metrics collection utilities."""

import asyncio
import math
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    execution_times: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    memory_usage: list[float] = field(default_factory=list)

    def add_execution(
        self, execution_time: float, tokens: int | None = None, success: bool = True
    ) -> None:
        """Add execution metrics."""
        self.execution_times.append(execution_time)
        if tokens is not None:
            self.token_counts.append(tokens)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def get_statistics(self) -> dict[str, Any]:
        """Calculate performance statistics."""
        stats = {}

        if self.execution_times:
            stats.update(
                {
                    "avg_execution_time": statistics.mean(self.execution_times),
                    "median_execution_time": statistics.median(self.execution_times),
                    "min_execution_time": min(self.execution_times),
                    "max_execution_time": max(self.execution_times),
                    "p95_execution_time": sorted(self.execution_times)[
                        math.ceil(0.95 * len(self.execution_times)) - 1
                    ],
                    "p99_execution_time": sorted(self.execution_times)[
                        math.ceil(0.99 * len(self.execution_times)) - 1
                    ],
                }
            )

        if self.token_counts:
            stats.update(
                {
                    "avg_tokens": statistics.mean(self.token_counts),
                    "total_tokens": sum(self.token_counts),
                    "max_tokens": max(self.token_counts),
                }
            )

        stats.update(
            {
                "total_executions": len(self.execution_times),
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": self.success_count
                / max(1, self.success_count + self.error_count),
            }
        )

        return stats


class PerformanceMonitor:
    """Global performance monitoring system."""

    def __init__(self) -> None:
        """Initialize the performance monitor."""
        self.metrics: dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.system_metrics = PerformanceMetrics()
        self.workflow_completion_times: list[float] = []
        self._lock = threading.Lock()

    def record_execution(
        self,
        component: str,
        execution_time: float,
        tokens: int | None = None,
        success: bool = True,
    ) -> None:
        """Record execution metrics for a component."""
        with self._lock:
            self.metrics[component].add_execution(execution_time, tokens, success)
            self.system_metrics.add_execution(execution_time, tokens, success)

    def record_token_usage(self, component: str, tokens: int) -> None:
        """Record token usage for a component."""
        with self._lock:
            self.metrics[component].token_counts.append(tokens)
            self.system_metrics.token_counts.append(tokens)

    def record_workflow_completion(self, completion_time: float) -> None:
        """Record workflow completion time."""
        with self._lock:
            self.workflow_completion_times.append(completion_time)

    @contextmanager
    def monitor_execution(
        self, component: str, tokens: int | None = None
    ) -> Iterator[None]:
        """Context manager for monitoring execution time."""
        start_time = time.time()
        success = True

        try:
            yield
        except Exception as e:
            success = False
            logger.error(f"Execution failed for {component}", error=str(e))
            raise
        finally:
            execution_time = time.time() - start_time
            self.record_execution(component, execution_time, tokens, success)

            if success:
                logger.debug(
                    f"Execution completed for {component}",
                    component=component,
                    execution_time=execution_time,
                    success=success,
                )

    def get_component_metrics(self, component: str) -> dict[str, Any]:
        """Get metrics for a specific component."""
        with self._lock:
            return self.metrics[component].get_statistics()

    def get_system_metrics(self) -> dict[str, Any]:
        """Get overall system metrics."""
        with self._lock:
            return self.system_metrics.get_statistics()

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all components."""
        with self._lock:
            return {
                "system": self.get_system_metrics(),
                "components": {
                    component: metrics.get_statistics()
                    for component, metrics in self.metrics.items()
                },
            }

    def get_statistics(self) -> dict[str, Any]:
        """Get combined statistics across all components with prefixed keys."""
        with self._lock:
            stats = {}

            # Add workflow completion metrics (separate from general execution times)
            if self.workflow_completion_times:
                stats["workflow_avg_completion"] = statistics.mean(
                    self.workflow_completion_times
                )
                stats["workflow_median_completion"] = statistics.median(
                    self.workflow_completion_times
                )

            # Add component-specific metrics with prefixed keys
            for component, metrics in self.metrics.items():
                component_stats = metrics.get_statistics()
                component_prefix = component.lower()

                # Map execution time stats
                if "avg_execution_time" in component_stats:
                    stats[f"{component_prefix}_avg_time"] = component_stats[
                        "avg_execution_time"
                    ]
                if "median_execution_time" in component_stats:
                    stats[f"{component_prefix}_median_time"] = component_stats[
                        "median_execution_time"
                    ]
                if "max_execution_time" in component_stats:
                    stats[f"{component_prefix}_max_time"] = component_stats[
                        "max_execution_time"
                    ]

                # Map token stats
                if "avg_tokens" in component_stats:
                    stats[f"{component_prefix}_avg_tokens"] = component_stats[
                        "avg_tokens"
                    ]
                if "total_tokens" in component_stats:
                    stats[f"{component_prefix}_total_tokens"] = component_stats[
                        "total_tokens"
                    ]

                # Map execution counts
                if "total_executions" in component_stats:
                    stats[f"{component_prefix}_executions"] = component_stats[
                        "total_executions"
                    ]
                if "success_rate" in component_stats:
                    stats[f"{component_prefix}_success_rate"] = component_stats[
                        "success_rate"
                    ]

            return stats

    def reset_metrics(self, component: str | None = None) -> None:
        """Reset metrics for a component or all components."""
        with self._lock:
            if component:
                self.metrics[component] = PerformanceMetrics()
            else:
                self.metrics.clear()
                self.system_metrics = PerformanceMetrics()
                self.workflow_completion_times.clear()

    def check_performance_thresholds(
        self, thresholds: dict[str, float]
    ) -> dict[str, bool]:
        """Check if components meet performance thresholds."""
        violations = {}

        with self._lock:
            for component, threshold in thresholds.items():
                if component in self.metrics:
                    stats = self.metrics[component].get_statistics()
                    avg_time = stats.get("avg_execution_time", 0)
                    violations[component] = avg_time > threshold

        return violations


# Global performance monitor instance
_performance_monitor = None


_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        with _monitor_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    return _performance_monitor


class AgentPerformanceTracker:
    """Performance tracker mixin for agents."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize performance tracking mixin."""
        super().__init__(*args, **kwargs)
        self.performance_monitor = get_performance_monitor()
        self.component_name = self.__class__.__name__

    @contextmanager
    def track_execution(self, tokens: int | None = None) -> Iterator[None]:
        """Track agent execution performance."""
        with self.performance_monitor.monitor_execution(self.component_name, tokens):
            yield

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for this agent."""
        return self.performance_monitor.get_component_metrics(self.component_name)


class WorkflowPerformanceTracker:
    """Performance tracker for workflow executions."""

    def __init__(self, workflow_name: str = "LL3M_Workflow"):
        """Initialize workflow performance tracker."""
        self.workflow_name = workflow_name
        self.performance_monitor = get_performance_monitor()
        self.step_times: dict[str, float] = {}
        self.workflow_start_time: float | None = None

    def start_workflow(self) -> None:
        """Start workflow timing."""
        self.workflow_start_time = time.time()
        self.step_times.clear()

    def record_step(
        self,
        step_name: str,
        execution_time: float,
        tokens: int | None = None,
        success: bool = True,
    ) -> None:
        """Record individual step performance."""
        self.step_times[step_name] = execution_time
        self.performance_monitor.record_execution(
            f"{self.workflow_name}_{step_name}", execution_time, tokens, success
        )

    def complete_workflow(self, success: bool = True) -> None:
        """Complete workflow timing."""
        if self.workflow_start_time is not None:
            total_time = time.time() - self.workflow_start_time
            total_tokens = sum(
                sum(
                    self.performance_monitor.metrics[
                        f"{self.workflow_name}_{step}"
                    ].token_counts
                )
                for step in self.step_times.keys()
                if f"{self.workflow_name}_{step}" in self.performance_monitor.metrics
                and self.performance_monitor.metrics[
                    f"{self.workflow_name}_{step}"
                ].token_counts
            )

            self.performance_monitor.record_execution(
                self.workflow_name,
                total_time,
                total_tokens if total_tokens else None,
                success,
            )

            logger.info(
                f"Workflow {self.workflow_name} completed",
                workflow=self.workflow_name,
                total_time=total_time,
                step_times=self.step_times,
                total_tokens=total_tokens,
                success=success,
            )

    def get_workflow_stats(self) -> dict[str, Any]:
        """Get workflow performance statistics."""
        return self.performance_monitor.get_component_metrics(self.workflow_name)


def monitor_function_performance(component_name: str) -> Any:
    """Decorator for monitoring function performance."""

    def decorator(func: Any) -> Any:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            monitor = get_performance_monitor()
            with monitor.monitor_execution(component_name):
                return await func(*args, **kwargs)

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            monitor = get_performance_monitor()
            with monitor.monitor_execution(component_name):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_performance_summary(interval_seconds: int = 300) -> None:
    """Log performance summary periodically."""
    import threading

    def log_summary() -> None:
        monitor = get_performance_monitor()
        metrics = monitor.get_all_metrics()

        logger.info(
            "Performance Summary",
            system_metrics=metrics["system"],
            component_count=len(metrics["components"]),
        )

        for component, stats in metrics["components"].items():
            if stats["total_executions"] > 0:
                logger.info(
                    f"Component Performance: {component}", component=component, **stats
                )

        # Schedule next summary
        threading.Timer(interval_seconds, log_summary).start()

    # Start periodic logging
    threading.Timer(interval_seconds, log_summary).start()


# Performance thresholds for different components
PERFORMANCE_THRESHOLDS = {
    "PlannerAgent": 2.0,  # 2 seconds for task planning
    "RetrievalAgent": 3.0,  # 3 seconds for documentation retrieval
    "CodingAgent": 2.5,  # 2.5 seconds for code generation
    "BlenderExecutor": 5.0,  # 5 seconds for Blender execution
    "LL3M_Workflow": 15.0,  # 15 seconds for full workflow
}


def check_performance_alerts() -> list[str]:
    """Check for performance threshold violations."""
    monitor = get_performance_monitor()
    violations = monitor.check_performance_thresholds(PERFORMANCE_THRESHOLDS)

    alerts = []
    for component, violated in violations.items():
        if violated:
            stats = monitor.get_component_metrics(component)
            avg_time = stats.get("avg_execution_time", 0)
            threshold = PERFORMANCE_THRESHOLDS[component]

            alert = (
                f"Performance Alert: {component} avg time {avg_time:.2f}s "
                f"exceeds threshold {threshold}s"
            )
            alerts.append(alert)
            logger.warning(
                alert, component=component, avg_time=avg_time, threshold=threshold
            )

    return alerts
