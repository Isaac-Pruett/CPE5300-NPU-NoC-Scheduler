"""Release-aware offline planning for one non-preemptive compute resource."""

from collections.abc import Callable

from schedulers.scheduler import Scheduler
from schedulers.task import Task
from speculative.event_model import EventTrace, EventType


def plan_single_processor(
    trace: EventTrace, policy_factory: Callable[[], Scheduler]
) -> EventTrace:
    """Select only arrived, dependency-ready jobs using one persistent policy.

    This planner uses known fixed durations, not live ASTRA feedback. ASTRA also
    enforces the releases independently. Integer nanoseconds match the pinned
    replay backend, including its 1 ns minimum for zero-duration compute nodes.
    Placement/completion here advances a planning clock, never Task.finished.
    """
    scheduled = EventTrace.from_dict(trace.to_dict())
    if scheduled.num_ranks != 1 or any(
        event.event_type is not EventType.COMPUTE for event in scheduled.events
    ):
        raise ValueError("release-aware planning requires one compute-only rank")
    by_id = {event.event_id: event for event in scheduled.events}
    original_dependencies = {
        event.event_id: set(event.dependencies) for event in scheduled.events
    }
    pending = set(by_id)
    queued = {}
    completed = set()
    policy = policy_factory()
    order = []
    now_ns = 0
    while pending or queued:
        eligible = sorted(
            (
                by_id[event_id]
                for event_id in pending
                if original_dependencies[event_id] <= completed
            ),
            key=lambda event: (event.arrival_us, event.event_id),
        )
        if not queued and eligible:
            now_ns = max(now_ns, eligible[0].arrival_us * 1000)
        arriving = [event for event in eligible if event.arrival_us * 1000 <= now_ns]
        if arriving:
            tasks = [Task(event=event) for event in arriving]
            policy.admit(tasks)
            for task in tasks:
                event_id = task.event.event_id
                queued[event_id] = task
                pending.remove(event_id)
        if not queued:
            raise ValueError(
                "no arrived dependency-ready work; single-processor dependency deadlock"
            )
        selected = policy.next()
        if (
            not isinstance(selected, Task)
            or queued.get(selected.event.event_id) is not selected
        ):
            raise ValueError(
                "scheduler must dequeue and return an admitted ready task exactly once"
            )
        event = selected.event
        queued.pop(event.event_id)
        if order and order[-1] not in event.dependencies:
            event.dependencies.append(order[-1])
        order.append(event.event_id)
        now_ns += max(1, event.duration_us * 1000)
        if now_ns > 2**64 - 1:
            raise ValueError(
                "planned completion exceeds the simulator's uint64 clock range"
            )
        completed.add(event.event_id)
    scheduled.parameters["compute_schedule"] = {"0": order}
    scheduled.parameters["scheduling_model"] = (
        "nonpreemptive_release_aware_fixed_duration"
    )
    scheduled.validate()
    return scheduled
