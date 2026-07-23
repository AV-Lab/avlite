"""TaskStrategy ABC, schedule/placement/event enums, and thin TaskRunner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy


class TaskSchedule(Enum):
    EVERY_CYCLE = auto()
    INTERVAL = auto()
    ON_EVENT = auto()


class TaskPlacement(Enum):
    """Where the task body runs - executer-agnostic; each executer maps as it can."""

    INLINE = auto()
    THREAD = auto()
    PROCESS = auto()


class StackEvent(Enum):
    EXECUTION_STARTED = auto()
    EXECUTION_STOPPED = auto()
    EXECUTION_RESET = auto()
    GOAL_ARRIVED = auto()
    PARKING_ZONE_ENTERED = auto()
    LOCAL_PLAN_FAILED = auto()
    LOCAL_PLAN_COLLISION = auto()
    LOCAL_PLAN_EXHAUSTED = auto()
    CONTROL_HALTED = auto()
    GLOBAL_PLAN_MISSING = auto()


class TaskStrategy(ABC):
    """Plugin-facing appendable behavior run by :class:`TaskRunner`."""

    registry: ClassVar[dict[str, type[TaskStrategy]]] = {}

    schedule: ClassVar[TaskSchedule] = TaskSchedule.EVERY_CYCLE
    placement: ClassVar[TaskPlacement] = TaskPlacement.INLINE
    interval_s: ClassVar[float] = 1.0 # For INTERVAL tasks, how often to run (seconds).
    listen_events: ClassVar[frozenset[StackEvent]] = frozenset()
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset()

    @abstractmethod
    def execute(
        self,
        executer: ExecutionStrategy,
        event: StackEvent | None = None,
    ) -> None:
        """Run one task invocation. Stack state is read from ``executer``."""

    def reset(self) -> None:
        """Clear task-owned state (e.g. rising-edge flags). Default no-op."""

    def __init_subclass__(cls, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            TaskStrategy.registry[cls.__name__] = cls


class TaskRunner:
    """Schedules CYCLE/INTERVAL tasks and dispatches ``notify`` to ON_EVENT listeners."""

    def __init__(
        self,
        tasks: list[TaskStrategy] | None = None,
        executer: ExecutionStrategy | None = None,
    ):
        self._tasks = list(tasks or [])
        self.executer = executer
        self._last_interval_fire: dict[int, float] = {}
        self._pending_events: list[StackEvent] = []
        self._in_step = False

    @property
    def tasks(self) -> list[TaskStrategy]:
        return self._tasks

    def reset(self) -> None:
        self._last_interval_fire.clear()
        self._pending_events.clear()
        for task in self._tasks:
            task.reset()

    def step(self, executer: ExecutionStrategy) -> None:
        """Run CYCLE/INTERVAL tasks, then flush any events they ``notify``d."""
        self._in_step = True
        try:
            for task in self._tasks:
                if task.schedule is TaskSchedule.EVERY_CYCLE:
                    executer.dispatch_task(task)
                elif task.schedule is TaskSchedule.INTERVAL:
                    key = id(task)
                    last = self._last_interval_fire.get(key)
                    now = float(executer.elapsed_sim_time)
                    if last is None or (now - last) >= float(task.interval_s):
                        self._last_interval_fire[key] = now
                        executer.dispatch_task(task)
            while self._pending_events:
                event = self._pending_events.pop(0)
                for task in self._tasks:
                    if task.schedule is TaskSchedule.ON_EVENT and event in task.listen_events:
                        executer.dispatch_task(task, event=event)
        finally:
            self._in_step = False

    def notify(
        self,
        event: StackEvent,
        executer: ExecutionStrategy | None = None,
    ) -> None:
        """Queue during :meth:`step`, otherwise dispatch ON_EVENT listeners immediately.

        Uses the bound :attr:`executer` when ``executer`` is omitted.
        """
        ex = executer if executer is not None else self.executer
        if ex is None:
            raise RuntimeError("TaskRunner.notify requires an executer (pass one or bind TaskRunner.executer)")
        if self._in_step:
            self._pending_events.append(event)
            return
        for task in self._tasks:
            if task.schedule is TaskSchedule.ON_EVENT and event in task.listen_events:
                ex.dispatch_task(task, event=event)
