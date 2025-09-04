"""Performance monitoring and optimization system."""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Callable
from collections.abc import Callable as CallableType
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import psutil
from pydantic import BaseModel


class PerformanceMetric(BaseModel):
    """Individual performance metric."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: dict[str, str] = {}


class PerformanceAlert(BaseModel):
    """Performance alert definition."""

    id: str
    name: str
    metric_name: str
    condition: str  # e.g., "> 80", "< 0.95"
    threshold: float
    severity: str  # "info", "warning", "error", "critical"
    description: str
    is_active: bool = True


class SystemMetrics(BaseModel):
    """System-level performance metrics."""

    cpu_usage: float
    memory_usage: float
    memory_available: int
    disk_usage: float
    disk_free: int
    load_average: list[float]
    network_io: dict[str, int]
    process_count: int


class ApplicationMetrics(BaseModel):
    """Application-level performance metrics."""

    active_connections: int
    requests_per_second: float
    average_response_time: float
    error_rate: float
    active_generations: int
    queue_depth: int
    cache_hit_rate: float


class BlenderMetrics(BaseModel):
    """Blender-specific performance metrics."""

    active_processes: int
    average_execution_time: float
    memory_per_process: float
    success_rate: float
    queue_length: int
    concurrent_limit: int


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self) -> None:
        self.metrics_buffer: dict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.alerts: dict[str, PerformanceAlert] = {}
        self.alert_callbacks: list[Callable[..., None]] = []
        self.monitoring_active = False

        # Performance counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times: deque[float] = deque(maxlen=100)

        # Caching for expensive operations
        self.cache: dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Resource optimization
        self.connection_pool_size = 20
        self.worker_pool_size = 4

        # Initialize default alerts
        self._setup_default_alerts()

    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        # Start background monitoring tasks
        asyncio.create_task(self._system_monitor(interval))
        asyncio.create_task(self._application_monitor(interval))
        asyncio.create_task(self._blender_monitor(interval))
        asyncio.create_task(self._alert_processor())
        asyncio.create_task(self._cache_cleaner())

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False

    async def _system_monitor(self, interval: float) -> None:
        """Monitor system-level metrics."""
        while self.monitoring_active:
            try:
                # CPU metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                self._record_metric("system.cpu.usage", cpu_usage, "percent")

                # Memory metrics
                memory = psutil.virtual_memory()
                self._record_metric("system.memory.usage", memory.percent, "percent")
                self._record_metric(
                    "system.memory.available", memory.available, "bytes"
                )

                # Disk metrics
                disk = psutil.disk_usage("/")
                disk_usage = (disk.used / disk.total) * 100
                self._record_metric("system.disk.usage", disk_usage, "percent")
                self._record_metric("system.disk.free", disk.free, "bytes")

                # Load average
                load_avg = psutil.getloadavg()
                self._record_metric("system.load.1min", load_avg[0], "load")
                self._record_metric("system.load.5min", load_avg[1], "load")
                self._record_metric("system.load.15min", load_avg[2], "load")

                # Network I/O
                net_io = psutil.net_io_counters()
                self._record_metric(
                    "system.network.bytes_sent", net_io.bytes_sent, "bytes"
                )
                self._record_metric(
                    "system.network.bytes_recv", net_io.bytes_recv, "bytes"
                )

                # Process count
                process_count = len(psutil.pids())
                self._record_metric("system.processes", process_count, "count")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"Error in system monitoring: {e}")
                await asyncio.sleep(interval)

    async def _application_monitor(self, interval: float) -> None:
        """Monitor application-level metrics."""
        while self.monitoring_active:
            try:
                # Request metrics
                rps = self.request_counter / interval
                self._record_metric("app.requests_per_second", rps, "rps")
                self.request_counter = 0

                # Error rate
                error_rate = (self.error_counter / max(rps * interval, 1)) * 100
                self._record_metric("app.error_rate", error_rate, "percent")
                self.error_counter = 0

                # Response time
                if self.response_times:
                    avg_response_time = sum(self.response_times) / len(
                        self.response_times
                    )
                    self._record_metric(
                        "app.response_time.avg", avg_response_time, "seconds"
                    )

                    # 95th percentile
                    sorted_times = sorted(self.response_times)
                    p95_index = int(len(sorted_times) * 0.95)
                    p95_time = sorted_times[p95_index] if sorted_times else 0
                    self._record_metric("app.response_time.p95", p95_time, "seconds")

                # Cache metrics
                total_cache_ops = self.cache_hits + self.cache_misses
                cache_hit_rate = (self.cache_hits / max(total_cache_ops, 1)) * 100
                self._record_metric("app.cache.hit_rate", cache_hit_rate, "percent")
                self._record_metric("app.cache.size", len(self.cache), "count")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"Error in application monitoring: {e}")
                await asyncio.sleep(interval)

    async def _blender_monitor(self, interval: float) -> None:
        """Monitor Blender-specific metrics."""
        while self.monitoring_active:
            try:
                # Count Blender processes
                blender_processes = []
                for proc in psutil.process_iter(
                    ["pid", "name", "memory_info", "create_time"]
                ):
                    try:
                        if "blender" in proc.info["name"].lower():
                            blender_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Active Blender processes
                active_processes = len(blender_processes)
                self._record_metric(
                    "blender.active_processes", active_processes, "count"
                )

                # Memory per process
                if blender_processes:
                    total_memory = sum(
                        proc.info["memory_info"].rss for proc in blender_processes
                    )
                    avg_memory = total_memory / len(blender_processes)
                    self._record_metric(
                        "blender.memory_per_process", avg_memory, "bytes"
                    )

                # Mock queue metrics (would come from actual queue system)
                queue_length = max(0, 15 - active_processes)  # Mock calculation
                self._record_metric("blender.queue_length", queue_length, "count")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"Error in Blender monitoring: {e}")
                await asyncio.sleep(interval)

    async def _alert_processor(self) -> None:
        """Process alerts based on current metrics."""
        while self.monitoring_active:
            try:
                for alert in self.alerts.values():
                    if not alert.is_active:
                        continue

                    # Get latest metric value
                    metric_values = self.metrics_buffer.get(alert.metric_name)
                    if not metric_values:
                        continue

                    latest_metric = metric_values[-1]

                    # Check alert condition
                    should_alert = self._evaluate_condition(
                        latest_metric.value, alert.condition, alert.threshold
                    )

                    if should_alert:
                        await self._trigger_alert(alert, latest_metric.value)

                await asyncio.sleep(10)  # Check alerts every 10 seconds

            except Exception as e:
                print(f"Error in alert processing: {e}")
                await asyncio.sleep(10)

    def _record_metric(
        self, name: str, value: float, unit: str, tags: dict[str, str] | None = None
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name, value=value, unit=unit, timestamp=datetime.now(), tags=tags or {}
        )
        self.metrics_buffer[name].append(metric)

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """Evaluate alert condition."""
        if condition.startswith(">"):
            return value > threshold
        elif condition.startswith("<"):
            return value < threshold
        elif condition.startswith(">="):
            return value >= threshold
        elif condition.startswith("<="):
            return value <= threshold
        elif condition.startswith("=="):
            return value == threshold
        elif condition.startswith("!="):
            return value != threshold
        return False

    async def _trigger_alert(
        self, alert: PerformanceAlert, current_value: float
    ) -> None:
        """Trigger a performance alert."""
        alert_data = {
            "id": alert.id,
            "name": alert.name,
            "metric_name": alert.metric_name,
            "current_value": current_value,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "description": alert.description,
            "timestamp": datetime.utcnow(),
        }

        print(f"ALERT: {alert.name} - {alert.description} (Current: {current_value})")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                print(f"Alert callback failed: {e}")

    def _setup_default_alerts(self) -> None:
        """Set up default performance alerts."""
        default_alerts = [
            PerformanceAlert(
                id="cpu_high",
                name="High CPU Usage",
                metric_name="system.cpu.usage",
                condition="> 80",
                threshold=80.0,
                severity="warning",
                description="CPU usage is above 80%",
            ),
            PerformanceAlert(
                id="memory_high",
                name="High Memory Usage",
                metric_name="system.memory.usage",
                condition="> 90",
                threshold=90.0,
                severity="error",
                description="Memory usage is above 90%",
            ),
            PerformanceAlert(
                id="disk_full",
                name="Disk Space Low",
                metric_name="system.disk.usage",
                condition="> 85",
                threshold=85.0,
                severity="warning",
                description="Disk usage is above 85%",
            ),
            PerformanceAlert(
                id="response_time_high",
                name="High Response Time",
                metric_name="app.response_time.avg",
                condition="> 2.0",
                threshold=2.0,
                severity="warning",
                description="Average response time is above 2 seconds",
            ),
            PerformanceAlert(
                id="error_rate_high",
                name="High Error Rate",
                metric_name="app.error_rate",
                condition="> 5.0",
                threshold=5.0,
                severity="error",
                description="Error rate is above 5%",
            ),
            PerformanceAlert(
                id="queue_backlog",
                name="Queue Backlog",
                metric_name="blender.queue_length",
                condition="> 50",
                threshold=50.0,
                severity="warning",
                description="Blender queue has more than 50 pending jobs",
            ),
        ]

        for alert in default_alerts:
            self.alerts[alert.id] = alert

    @asynccontextmanager
    async def track_request_time(self) -> AsyncGenerator[None, None]:
        """Context manager to track request execution time."""
        start_time = time.time()
        try:
            yield
            self.request_counter += 1
        except Exception:
            self.error_counter += 1
            raise
        finally:
            execution_time = time.time() - start_time
            self.response_times.append(execution_time)

    async def get_current_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        # System metrics
        try:
            system_metrics = SystemMetrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                memory_available=psutil.virtual_memory().available,
                disk_usage=(psutil.disk_usage("/").used / psutil.disk_usage("/").total)
                * 100,
                disk_free=psutil.disk_usage("/").free,
                load_average=list(psutil.getloadavg()),
                network_io={
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv,
                },
                process_count=len(psutil.pids()),
            )
        except Exception:
            system_metrics = None

        # Application metrics
        rps = len(self.response_times) / 60 if self.response_times else 0
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0
        )

        application_metrics = ApplicationMetrics(
            active_connections=25,  # Mock value
            requests_per_second=rps,
            average_response_time=avg_response_time,
            error_rate=(self.error_counter / max(self.request_counter, 1)) * 100,
            active_generations=8,  # Mock value
            queue_depth=12,  # Mock value
            cache_hit_rate=(
                self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            )
            * 100,
        )

        # Blender metrics
        blender_processes = 0
        try:
            for proc in psutil.process_iter(["name"]):
                if "blender" in proc.info["name"].lower():
                    blender_processes += 1
        except Exception as e:  # nosec B110
            # Ignore process enumeration errors - they're not critical for monitoring
            print(f"Process enumeration error (non-critical): {e}")  # Simple logging for now

        blender_metrics = BlenderMetrics(
            active_processes=blender_processes,
            average_execution_time=180.5,  # Mock value
            memory_per_process=2048 * 1024 * 1024,  # 2GB mock value
            success_rate=94.2,  # Mock value
            queue_length=max(0, 15 - blender_processes),
            concurrent_limit=5,
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics.model_dump() if system_metrics else None,
            "application": application_metrics.model_dump(),
            "blender": blender_metrics.model_dump(),
        }

    async def get_metric_history(
        self, metric_name: str, _duration: timedelta = timedelta(hours=1)
    ) -> list[float]:
        """Get historical data for a specific metric."""
        metrics = list(self.metrics_buffer.get(metric_name, []))

        # Extract float values from PerformanceMetric objects
        values = [metric.value for metric in metrics[-100:]] if metrics else []
        return values

    def add_alert_callback(self, callback: Callable[..., None]) -> None:
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def create_alert(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str = "warning",
        description: str = "",
    ) -> str:
        """Create a new performance alert."""
        alert_id = str(uuid4())

        alert = PerformanceAlert(
            id=alert_id,
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            description=description,
        )

        self.alerts[alert_id] = alert
        return alert_id

    def delete_alert(self, alert_id: str) -> bool:
        """Delete a performance alert."""
        return self.alerts.pop(alert_id, None) is not None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_ops = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_ops * 100) if total_ops > 0 else 0

        return {
            "size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_operations": total_ops,
        }

    def cached_operation(self, key: str, ttl: int = 300) -> CallableType[..., Any]:
        """Decorator for caching expensive operations."""

        def decorator(func: CallableType[..., Any]) -> CallableType[..., Any]:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                cache_key = f"{key}_{hash(str(args) + str(kwargs))}"

                # Check cache
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if datetime.utcnow() - timestamp < timedelta(seconds=ttl):
                        self.cache_hits += 1
                        return cached_data

                # Cache miss - execute function
                self.cache_misses += 1
                result = await func(*args, **kwargs)

                # Store in cache
                self.cache[cache_key] = (result, datetime.utcnow())
                return result

            return wrapper

        return decorator

    async def _cache_cleaner(self) -> None:
        """Clean expired cache entries."""
        while self.monitoring_active:
            try:
                now = datetime.utcnow()
                expired_keys = []

                for key, (_value, timestamp) in self.cache.items():
                    if now - timestamp > timedelta(
                        seconds=600
                    ):  # 10 minutes default TTL
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.cache[key]

                await asyncio.sleep(300)  # Clean every 5 minutes

            except Exception as e:
                print(f"Error in cache cleaning: {e}")
                await asyncio.sleep(300)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
