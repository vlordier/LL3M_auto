"""Advanced batch processing system for LL3M."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from ..api.models import GenerateAssetRequest


class BatchStatus(str, Enum):
    """Batch processing status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BatchPriority(int, Enum):
    """Batch priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class BatchJobItem(BaseModel):
    """Individual item in a batch job."""

    id: UUID
    request: GenerateAssetRequest
    status: BatchStatus = BatchStatus.QUEUED
    asset_id: UUID | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    execution_time: float | None = None


class BatchJob(BaseModel):
    """Batch processing job."""

    id: UUID
    name: str
    user_id: UUID
    items: list[BatchJobItem]
    status: BatchStatus = BatchStatus.QUEUED
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimated_completion: datetime | None = None
    progress: float = 0.0

    # Configuration
    max_retries: int = 3
    parallel_workers: int = 3
    timeout_per_item: int = 600  # seconds
    notify_on_completion: bool = True

    # Statistics
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0

    def __post_init__(self):
        self.total_items = len(self.items)


class BatchConfiguration(BaseModel):
    """Configuration for batch processing."""

    max_parallel_jobs: int = 5
    max_items_per_job: int = 100
    default_timeout: int = 600
    max_retries: int = 3
    queue_cleanup_interval: int = 3600  # seconds


class BatchProcessor:
    """Advanced batch processing system."""

    def __init__(self, config: BatchConfiguration | None = None):
        self.config = config or BatchConfiguration()
        self.active_jobs: dict[UUID, BatchJob] = {}
        self.job_queue: list[UUID] = []
        self.processing_locks: dict[UUID, asyncio.Lock] = {}
        self.workflow_graph = None
        self.notification_callbacks: list[Callable] = []
        self._shutdown_event = asyncio.Event()
        self._background_tasks: set[asyncio.Task] = set()

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        processor_task = asyncio.create_task(self._job_processor())
        cleaner_task = asyncio.create_task(self._queue_cleaner())

        self._background_tasks.add(processor_task)
        self._background_tasks.add(cleaner_task)

        # Clean up finished tasks
        processor_task.add_done_callback(self._background_tasks.discard)
        cleaner_task.add_done_callback(self._background_tasks.discard)

    async def shutdown(self) -> None:
        """Gracefully shutdown the batch processor."""
        self._shutdown_event.set()

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Cancel any active jobs
        for job in self.active_jobs.values():
            if job.status == BatchStatus.PROCESSING:
                job.status = BatchStatus.FAILED
                job.completed_at = datetime.utcnow()

    async def __aenter__(self) -> "BatchProcessor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.shutdown()

    async def create_batch_job(
        self,
        name: str,
        user_id: UUID,
        requests: list[GenerateAssetRequest],
        priority: BatchPriority = BatchPriority.NORMAL,
        **kwargs,
    ) -> BatchJob:
        """Create a new batch processing job."""
        if len(requests) > self.config.max_items_per_job:
            raise ValueError(
                f"Too many items. Maximum allowed: {self.config.max_items_per_job}"
            )

        # Create batch job
        job_id = uuid4()
        items = [BatchJobItem(id=uuid4(), request=request) for request in requests]

        batch_job = BatchJob(
            id=job_id,
            name=name,
            user_id=user_id,
            items=items,
            priority=priority,
            created_at=datetime.utcnow(),
            **kwargs,
        )

        # Estimate completion time
        avg_time_per_item = 180  # seconds (would be calculated from historical data)
        parallel_factor = min(batch_job.parallel_workers, len(items))
        estimated_duration = (len(items) * avg_time_per_item) / parallel_factor

        batch_job.estimated_completion = datetime.utcnow() + timedelta(
            seconds=estimated_duration
        )

        # Add to queue
        self.active_jobs[job_id] = batch_job
        self._insert_job_by_priority(job_id)

        return batch_job

    def _insert_job_by_priority(self, job_id: UUID) -> None:
        """Insert job into queue based on priority."""
        job = self.active_jobs[job_id]

        # Find insertion point based on priority
        insertion_index = 0
        for i, queued_job_id in enumerate(self.job_queue):
            queued_job = self.active_jobs[queued_job_id]
            if job.priority.value > queued_job.priority.value:
                insertion_index = i
                break
            insertion_index = i + 1

        self.job_queue.insert(insertion_index, job_id)

    async def _job_processor(self) -> None:
        """Background task to process batch jobs."""
        while not self._shutdown_event.is_set():
            try:
                # Check if we can process more jobs
                active_processing_jobs = sum(
                    1
                    for job in self.active_jobs.values()
                    if job.status == BatchStatus.PROCESSING
                )

                if active_processing_jobs >= self.config.max_parallel_jobs:
                    await asyncio.sleep(5)
                    continue

                # Get next job from queue
                if not self.job_queue:
                    await asyncio.sleep(10)
                    continue

                job_id = self.job_queue.pop(0)
                job = self.active_jobs.get(job_id)

                if not job or job.status != BatchStatus.QUEUED:
                    continue

                # Start processing job
                asyncio.create_task(self._process_batch_job(job))

            except Exception as e:
                print(f"Error in job processor: {e}")
                await asyncio.sleep(5)

    async def _process_batch_job(self, job: BatchJob) -> None:
        """Process a single batch job."""
        try:
            # Update job status
            job.status = BatchStatus.PROCESSING
            job.started_at = datetime.utcnow()

            # Create processing lock
            self.processing_locks[job.id] = asyncio.Lock()

            # Process items in parallel
            semaphore = asyncio.Semaphore(job.parallel_workers)
            tasks = [
                self._process_batch_item(job, item, semaphore)
                for item in job.items
                if item.status == BatchStatus.QUEUED
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Update job completion status
            job.completed_at = datetime.utcnow()
            job.progress = 100.0

            # Determine final status
            if job.failed_items == 0:
                job.status = BatchStatus.COMPLETED
            elif job.completed_items > 0:
                job.status = BatchStatus.COMPLETED  # Partial success
            else:
                job.status = BatchStatus.FAILED

            # Send notifications
            if job.notify_on_completion:
                await self._send_batch_notification(job)

        except Exception as e:
            job.status = BatchStatus.FAILED
            job.completed_at = datetime.utcnow()
            print(f"Batch job {job.id} failed: {e}")

        finally:
            # Cleanup
            if job.id in self.processing_locks:
                del self.processing_locks[job.id]

    async def _process_batch_item(
        self, job: BatchJob, item: BatchJobItem, semaphore: asyncio.Semaphore
    ) -> None:
        """Process a single item in a batch job."""
        async with semaphore:
            try:
                # Check if job was cancelled
                if job.status == BatchStatus.CANCELLED:
                    return

                # Update item status
                item.status = BatchStatus.PROCESSING
                item.started_at = datetime.utcnow()

                # Simulate asset generation (would use actual workflow)
                start_time = datetime.utcnow()

                # Mock asset generation
                await asyncio.sleep(2)  # Simulate processing time

                # Create mock asset
                asset_id = uuid4()
                item.asset_id = asset_id
                item.status = BatchStatus.COMPLETED
                item.completed_at = datetime.utcnow()
                item.execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Update job progress
                async with self.processing_locks[job.id]:
                    job.completed_items += 1
                    job.progress = (job.completed_items / job.total_items) * 100

            except Exception as e:
                # Handle item failure
                item.error_message = str(e)
                item.retry_count += 1

                if item.retry_count < job.max_retries:
                    # Retry the item
                    item.status = BatchStatus.QUEUED
                    await asyncio.sleep(5)  # Brief delay before retry
                    await self._process_batch_item(job, item, semaphore)
                else:
                    # Max retries exceeded
                    item.status = BatchStatus.FAILED
                    item.completed_at = datetime.utcnow()

                    async with self.processing_locks[job.id]:
                        job.failed_items += 1
                        job.progress = (
                            (job.completed_items + job.failed_items) / job.total_items
                        ) * 100

    async def get_batch_job(self, job_id: UUID) -> BatchJob | None:
        """Get batch job by ID."""
        return self.active_jobs.get(job_id)

    async def get_user_batch_jobs(self, user_id: UUID) -> list[BatchJob]:
        """Get all batch jobs for a user."""
        return [job for job in self.active_jobs.values() if job.user_id == user_id]

    async def cancel_batch_job(self, job_id: UUID) -> bool:
        """Cancel a batch job."""
        job = self.active_jobs.get(job_id)

        if not job:
            return False

        if job.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
        ]:
            return False

        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        # Cancel individual items
        for item in job.items:
            if item.status in [BatchStatus.QUEUED, BatchStatus.PROCESSING]:
                item.status = BatchStatus.CANCELLED

        return True

    async def pause_batch_job(self, job_id: UUID) -> bool:
        """Pause a batch job."""
        job = self.active_jobs.get(job_id)

        if not job or job.status != BatchStatus.PROCESSING:
            return False

        job.status = BatchStatus.PAUSED
        return True

    async def resume_batch_job(self, job_id: UUID) -> bool:
        """Resume a paused batch job."""
        job = self.active_jobs.get(job_id)

        if not job or job.status != BatchStatus.PAUSED:
            return False

        job.status = BatchStatus.QUEUED
        self._insert_job_by_priority(job_id)
        return True

    async def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        queued_jobs = len(
            [j for j in self.active_jobs.values() if j.status == BatchStatus.QUEUED]
        )
        processing_jobs = len(
            [j for j in self.active_jobs.values() if j.status == BatchStatus.PROCESSING]
        )

        return {
            "total_jobs": len(self.active_jobs),
            "queued_jobs": queued_jobs,
            "processing_jobs": processing_jobs,
            "queue_length": len(self.job_queue),
            "max_parallel_jobs": self.config.max_parallel_jobs,
            "average_processing_time": self._calculate_average_processing_time(),
        }

    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time for completed jobs."""
        completed_jobs = [
            job
            for job in self.active_jobs.values()
            if job.status == BatchStatus.COMPLETED
            and job.started_at
            and job.completed_at
        ]

        if not completed_jobs:
            return 0.0

        total_time = sum(
            (job.completed_at - job.started_at).total_seconds()
            for job in completed_jobs
            if job.started_at is not None and job.completed_at is not None
        )

        return total_time / len(completed_jobs)

    async def _queue_cleaner(self) -> None:
        """Background task to clean up old completed jobs."""
        while not self._shutdown_event.is_set():
            try:
                cleanup_threshold = datetime.utcnow() - timedelta(hours=24)

                # Remove old completed jobs
                jobs_to_remove = [
                    job_id
                    for job_id, job in self.active_jobs.items()
                    if job.status
                    in [
                        BatchStatus.COMPLETED,
                        BatchStatus.FAILED,
                        BatchStatus.CANCELLED,
                    ]
                    and job.completed_at
                    and job.completed_at < cleanup_threshold
                ]

                for job_id in jobs_to_remove:
                    del self.active_jobs[job_id]

                # Remove from queue if still there
                self.job_queue = [
                    job_id for job_id in self.job_queue if job_id not in jobs_to_remove
                ]

                await asyncio.sleep(self.config.queue_cleanup_interval)

            except Exception as e:
                print(f"Error in queue cleaner: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def _send_batch_notification(self, job: BatchJob) -> None:
        """Send notification when batch job completes."""
        notification_data = {
            "job_id": str(job.id),
            "job_name": job.name,
            "user_id": str(job.user_id),
            "status": job.status,
            "total_items": job.total_items,
            "completed_items": job.completed_items,
            "failed_items": job.failed_items,
            "execution_time": (job.completed_at - job.started_at).total_seconds()
            if job.completed_at and job.started_at
            else None,
        }

        # Call notification callbacks
        for callback in self.notification_callbacks:
            try:
                await callback(notification_data)
            except Exception as e:
                print(f"Notification callback failed: {e}")

    def add_notification_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Add callback for batch completion notifications."""
        self.notification_callbacks.append(callback)

    async def get_batch_statistics(self) -> dict[str, Any]:
        """Get comprehensive batch processing statistics."""
        total_jobs = len(self.active_jobs)

        status_counts = {}
        for status in BatchStatus:
            status_counts[status.value] = len(
                [job for job in self.active_jobs.values() if job.status == status]
            )

        # Calculate success rates
        completed_jobs = [
            j for j in self.active_jobs.values() if j.status == BatchStatus.COMPLETED
        ]
        total_items = sum(job.total_items for job in completed_jobs)
        successful_items = sum(job.completed_items for job in completed_jobs)

        success_rate = (successful_items / total_items * 100) if total_items > 0 else 0

        return {
            "total_jobs": total_jobs,
            "status_distribution": status_counts,
            "average_processing_time": self._calculate_average_processing_time(),
            "success_rate": success_rate,
            "total_items_processed": total_items,
            "successful_items": successful_items,
            "queue_efficiency": len(self.job_queue) / max(total_jobs, 1),
        }


# Global batch processor instance
batch_processor = BatchProcessor()
