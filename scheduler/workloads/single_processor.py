"""Independent arrival-zero jobs on one non-preemptive compute resource."""

import random

from speculative.event_model import Event, EventTrace, EventType


def build_single_processor_demo(jobs: int | None = None, seed: int = 42) -> EventTrace:
    """Keep the fixed regression, or generate a seeded longer arrival-zero batch."""
    if jobs is None:
        durations = [1000, 100, 10]
        names = ["long_job", "medium_job", "short_job"]
    else:
        if type(jobs) is not int or jobs < 1:
            raise ValueError("jobs must be a positive integer")
        rng = random.Random(seed)
        durations = [rng.randint(10, 1000) for _ in range(jobs)]
        names = [f"job_{index}" for index in range(jobs)]
    trace = EventTrace(
        num_ranks=1,
        events=[
            Event(index, EventType.COMPUTE, 0, name, duration_us=duration)
            for index, (name, duration) in enumerate(zip(names, durations))
        ],
        parameters={
            "workload": "single_processor_demo",
            "timing_source": "synthetic",
            "job_durations_us": durations,
            "arrival_us": 0,
            "scheduling_overhead_us": 0,
            "compute_resources": 1,
        },
    )
    if jobs is not None:
        trace.parameters.update(
            {
                "jobs": jobs,
                "seed": seed,
                "duration_distribution": "uniform_integer",
                "min_duration_us": 10,
                "max_duration_us": 1000,
            }
        )
    trace.validate()
    return trace


def single_processor_metrics(trace: EventTrace) -> dict:
    """Analytical latency for this demo, not ASTRA completion timestamp extraction.

    All jobs arrive at zero, are independent before policy serialization, and
    execute without idle time or context-switch overhead. This is deliberately
    restricted to the demo; it is not a general DAG/network timing model.
    """
    trace.validate()
    if (
        trace.num_ranks != 1
        or trace.parameters.get("workload") != "single_processor_demo"
    ):
        raise ValueError(
            "analytical latency is only supported for single-processor-demo"
        )
    if any(event.event_type is not EventType.COMPUTE for event in trace.events):
        raise ValueError("single-processor latency requires compute-only events")
    schedule = trace.parameters.get("compute_schedule", {})
    order = schedule.get("0", [])
    by_id = {event.event_id: event for event in trace.events}
    if len(order) != len(by_id) or set(order) != set(by_id):
        raise ValueError(
            "single-processor latency requires a complete explicit compute schedule"
        )
    elapsed = 0
    placed = set()
    tasks = []
    for event_id in order:
        event = by_id[event_id]
        if not set(event.dependencies) <= placed:
            raise ValueError("compute order violates event dependencies")
        start = elapsed
        elapsed += event.duration_us
        tasks.append(
            {
                "event_id": event_id,
                "name": event.name,
                "duration_us": event.duration_us,
                "start_us": start,
                "completion_us": elapsed,
                "waiting_us": start,
            }
        )
        placed.add(event_id)
    return {
        "source": "analytical_from_trace_durations_not_astra_timestamps",
        "mean_completion_us": sum(task["completion_us"] for task in tasks) / len(tasks),
        "mean_waiting_us": sum(task["waiting_us"] for task in tasks) / len(tasks),
        "makespan_us": elapsed,
        "tasks": tasks,
    }
