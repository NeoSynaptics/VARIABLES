"""
TaskQueue — Async FIFO task queue with pause/resume and priority support.

Usage:
    queue = TaskQueue(concurrency=3)
    await queue.start()
    task_id = await queue.submit(my_coroutine, priority=1)
    result = await queue.wait(task_id)
    await queue.stop()
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskItem:
    id: str
    fn: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class TaskQueue:
    """Async FIFO task queue with concurrency control, pause/resume, and priority."""

    def __init__(self, concurrency: int = 1, on_complete: Callable = None, on_error: Callable = None):
        self.concurrency = concurrency
        self.on_complete = on_complete
        self.on_error = on_error
        self._tasks: dict[str, TaskItem] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._running = False
        self._completions: dict[str, asyncio.Event] = {}

    async def start(self):
        """Start worker tasks."""
        self._running = True
        for i in range(self.concurrency):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

    async def stop(self, cancel_pending: bool = False):
        """Stop all workers."""
        self._running = False
        if cancel_pending:
            for tid, task in self._tasks.items():
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
        self._pause_event.set()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, fn: Callable[..., Coroutine], *args, priority: int = 0, **kwargs) -> str:
        """Submit a task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        item = TaskItem(id=task_id, fn=fn, args=args, kwargs=kwargs, priority=priority)
        self._tasks[task_id] = item
        self._completions[task_id] = asyncio.Event()
        await self._queue.put((priority, item.created_at, task_id))
        return task_id

    async def wait(self, task_id: str, timeout: float = None) -> Any:
        """Wait for a task to complete and return its result."""
        if task_id not in self._completions:
            raise KeyError(f"Unknown task: {task_id}")
        await asyncio.wait_for(self._completions[task_id].wait(), timeout=timeout)
        task = self._tasks[task_id]
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(task.error)
        return task.result

    def pause(self):
        """Pause processing (running tasks finish, new ones wait)."""
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        """Resume processing."""
        self._paused = False
        self._pause_event.set()

    def status(self, task_id: str = None) -> dict:
        """Get queue or task status."""
        if task_id:
            t = self._tasks.get(task_id)
            if not t:
                return {"error": "not found"}
            return {"id": t.id, "status": t.status.value, "priority": t.priority,
                    "error": t.error, "latency_ms": round((t.completed_at - t.started_at) * 1000, 1)
                    if t.completed_at and t.started_at else None}
        counts = {}
        for t in self._tasks.values():
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return {"total": len(self._tasks), "paused": self._paused, "concurrency": self.concurrency, **counts}

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            if task_id in self._completions:
                self._completions[task_id].set()
            return True
        return False

    async def _worker(self, name: str):
        while self._running:
            try:
                await self._pause_event.wait()
                priority, created, task_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                task = self._tasks.get(task_id)
                if not task or task.status == TaskStatus.CANCELLED:
                    continue
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                try:
                    task.result = await task.fn(*task.args, **task.kwargs)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    if self.on_complete:
                        self.on_complete(task)
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
                    if self.on_error:
                        self.on_error(task, e)
                finally:
                    if task_id in self._completions:
                        self._completions[task_id].set()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
