"""Tests for performance monitoring utilities."""

from unittest.mock import patch

import pytest

from src.utils.monitoring import (
    AgentPerformanceTracker,
    PerformanceMetrics,
    PerformanceMonitor,
    WorkflowPerformanceTracker,
    check_performance_alerts,
    get_performance_monitor,
    monitor_function_performance,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics data structure."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        metrics = PerformanceMetrics()
        assert metrics.execution_times == []
        assert metrics.token_counts == []
        assert metrics.error_count == 0
        assert metrics.success_count == 0
        assert metrics.memory_usage == []

    def test_add_execution_success(self):
        """Test adding successful execution."""
        metrics = PerformanceMetrics()
        metrics.add_execution(1.5, 100, True)

        assert metrics.execution_times == [1.5]
        assert metrics.token_counts == [100]
        assert metrics.success_count == 1
        assert metrics.error_count == 0

    def test_add_execution_failure(self):
        """Test adding failed execution."""
        metrics = PerformanceMetrics()
        metrics.add_execution(2.0, 50, False)

        assert metrics.execution_times == [2.0]
        assert metrics.token_counts == [50]
        assert metrics.success_count == 0
        assert metrics.error_count == 1

    def test_add_execution_no_tokens(self):
        """Test adding execution without token count."""
        metrics = PerformanceMetrics()
        metrics.add_execution(1.0, None, True)

        assert metrics.execution_times == [1.0]
        assert metrics.token_counts == []
        assert metrics.success_count == 1

    def test_get_statistics_empty(self):
        """Test statistics with no data."""
        metrics = PerformanceMetrics()
        stats = metrics.get_statistics()

        assert stats["total_executions"] == 0
        assert stats["success_count"] == 0
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_statistics_with_data(self):
        """Test statistics with execution data."""
        metrics = PerformanceMetrics()
        metrics.add_execution(1.0, 100, True)
        metrics.add_execution(2.0, 200, True)
        metrics.add_execution(3.0, 150, False)

        stats = metrics.get_statistics()

        assert stats["total_executions"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["avg_execution_time"] == 2.0
        assert stats["median_execution_time"] == 2.0
        assert stats["min_execution_time"] == 1.0
        assert stats["max_execution_time"] == 3.0
        assert stats["avg_tokens"] == 150.0
        assert stats["total_tokens"] == 450
        assert stats["max_tokens"] == 200


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert len(monitor.metrics) == 0
        assert monitor.system_metrics is not None

    def test_record_execution(self):
        """Test recording execution metrics."""
        monitor = PerformanceMonitor()
        monitor.record_execution("test_component", 1.5, 100, True)

        component_metrics = monitor.get_component_metrics("test_component")
        assert component_metrics["total_executions"] == 1
        assert component_metrics["avg_execution_time"] == 1.5

        system_metrics = monitor.get_system_metrics()
        assert system_metrics["total_executions"] == 1

    def test_monitor_execution_context_success(self):
        """Test monitor execution context manager - success."""
        monitor = PerformanceMonitor()

        with monitor.monitor_execution("test_component"):
            pass  # Quick execution

        stats = monitor.get_component_metrics("test_component")
        assert stats["total_executions"] == 1
        assert stats["success_count"] == 1
        assert stats["error_count"] == 0
        assert stats["avg_execution_time"] >= 0

    def test_monitor_execution_context_failure(self):
        """Test monitor execution context manager - failure."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            with monitor.monitor_execution("test_component"):
                raise ValueError("Test error")

        stats = monitor.get_component_metrics("test_component")
        assert stats["total_executions"] == 1
        assert stats["success_count"] == 0
        assert stats["error_count"] == 1

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        monitor = PerformanceMonitor()
        monitor.record_execution("component1", 1.0, 100, True)
        monitor.record_execution("component2", 2.0, 200, True)

        all_metrics = monitor.get_all_metrics()

        assert "system" in all_metrics
        assert "components" in all_metrics
        assert "component1" in all_metrics["components"]
        assert "component2" in all_metrics["components"]

    def test_reset_metrics_specific_component(self):
        """Test resetting specific component metrics."""
        monitor = PerformanceMonitor()
        monitor.record_execution("component1", 1.0, 100, True)
        monitor.record_execution("component2", 2.0, 200, True)

        monitor.reset_metrics("component1")

        stats1 = monitor.get_component_metrics("component1")
        stats2 = monitor.get_component_metrics("component2")

        assert stats1["total_executions"] == 0
        assert stats2["total_executions"] == 1

    def test_reset_metrics_all(self):
        """Test resetting all metrics."""
        monitor = PerformanceMonitor()
        monitor.record_execution("component1", 1.0, 100, True)

        monitor.reset_metrics()

        stats = monitor.get_component_metrics("component1")
        system_stats = monitor.get_system_metrics()

        assert stats["total_executions"] == 0
        assert system_stats["total_executions"] == 0

    def test_check_performance_thresholds(self):
        """Test performance threshold checking."""
        monitor = PerformanceMonitor()
        monitor.record_execution("fast_component", 0.5, 50, True)
        monitor.record_execution("slow_component", 5.0, 100, True)

        thresholds = {"fast_component": 1.0, "slow_component": 2.0}
        violations = monitor.check_performance_thresholds(thresholds)

        assert violations["fast_component"] is False
        assert violations["slow_component"] is True


class TestGlobalPerformanceMonitor:
    """Test global performance monitor functions."""

    def test_get_performance_monitor_singleton(self):
        """Test that get_performance_monitor returns singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        assert monitor1 is monitor2

    def test_check_performance_alerts(self):
        """Test performance alerts checking."""
        monitor = get_performance_monitor()
        monitor.reset_metrics()  # Clean slate

        # Add slow execution
        monitor.record_execution("PlannerAgent", 3.0, 100, True)

        with patch("src.utils.monitoring.logger") as mock_logger:
            alerts = check_performance_alerts()

            assert len(alerts) == 1
            assert "PlannerAgent" in alerts[0]
            assert "3.00s" in alerts[0]
            mock_logger.warning.assert_called_once()


class TestAgentPerformanceTracker:
    """Test AgentPerformanceTracker mixin."""

    def test_initialization(self):
        """Test tracker initialization."""

        class TestAgent(AgentPerformanceTracker):
            def __init__(self):
                super().__init__()

        agent = TestAgent()
        assert agent.component_name == "TestAgent"
        assert agent.performance_monitor is not None

    def test_track_execution_success(self):
        """Test successful execution tracking."""

        class TestAgent(AgentPerformanceTracker):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        with agent.track_execution(tokens=100):
            pass  # Quick execution

        stats = agent.get_performance_stats()
        assert stats["total_executions"] == 1
        assert stats["success_count"] == 1

    def test_track_execution_failure(self):
        """Test failed execution tracking."""

        class TestAgent(AgentPerformanceTracker):
            def __init__(self):
                super().__init__()

        agent = TestAgent()

        with pytest.raises(ValueError):
            with agent.track_execution():
                raise ValueError("Test error")

        stats = agent.get_performance_stats()
        assert stats["total_executions"] == 1
        assert stats["error_count"] == 1


class TestWorkflowPerformanceTracker:
    """Test WorkflowPerformanceTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = WorkflowPerformanceTracker("test_workflow")
        assert tracker.workflow_name == "test_workflow"
        assert tracker.step_times == {}
        assert tracker.workflow_start_time is None

    def test_workflow_lifecycle(self):
        """Test complete workflow lifecycle."""
        tracker = WorkflowPerformanceTracker("test_workflow")

        tracker.start_workflow()
        assert tracker.workflow_start_time is not None

        tracker.record_step("step1", 1.0, 50, True)
        tracker.record_step("step2", 2.0, 100, True)

        assert tracker.step_times["step1"] == 1.0
        assert tracker.step_times["step2"] == 2.0

        with patch("src.utils.monitoring.logger") as mock_logger:
            tracker.complete_workflow(True)
            mock_logger.info.assert_called_once()

    def test_get_workflow_stats(self):
        """Test getting workflow statistics."""
        tracker = WorkflowPerformanceTracker("test_workflow")
        tracker.start_workflow()
        tracker.record_step("step1", 1.0, 50, True)
        tracker.complete_workflow(True)

        stats = tracker.get_workflow_stats()
        assert "total_executions" in stats


class TestMonitorFunctionDecorator:
    """Test monitor_function_performance decorator."""

    def test_sync_function_monitoring(self):
        """Test monitoring synchronous function."""

        @monitor_function_performance("test_function")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

        monitor = get_performance_monitor()
        stats = monitor.get_component_metrics("test_function")
        assert stats["total_executions"] >= 1

    @pytest.mark.asyncio
    async def test_async_function_monitoring(self):
        """Test monitoring asynchronous function."""

        @monitor_function_performance("test_async_function")
        async def test_async_func():
            return "async_success"

        result = await test_async_func()
        assert result == "async_success"

        monitor = get_performance_monitor()
        stats = monitor.get_component_metrics("test_async_function")
        assert stats["total_executions"] >= 1
