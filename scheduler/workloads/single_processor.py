"""Independent jobs with release times on one non-preemptive compute resource."""

import random

from speculative.event_model import Event, EventTrace, EventType

MAX_ARRIVAL_US = (2**64 - 1) // 1000


def build_single_processor_demo(
    jobs: int | None = None,
    seed: int = 42,
    *,
    arrival_interval_us: int = 0,
    arrival_times_us: list[int] | None = None,
) -> EventTrace:
    """Keep the fixed jobs or generate seeded jobs, with optional release times.

    Explicit arrivals are in job order, not sorted by release time. Release times
    must fit an unsigned 64-bit nanosecond timestamp after conversion from us.
    """
    if (
        type(arrival_interval_us) is not int
        or not 0 <= arrival_interval_us <= MAX_ARRIVAL_US
    ):
        raise ValueError(
            f"arrival_interval_us must be a non-negative integer <= {MAX_ARRIVAL_US}"
        )
    if arrival_times_us is not None:
        if type(arrival_times_us) is not list:
            raise ValueError("arrival_times_us must be a list of integer microseconds")
        if arrival_interval_us != 0:
            raise ValueError(
                "arrival_times_us cannot be combined with a nonzero interval"
            )
    if jobs is None:
        durations = [1000, 100, 10]
        names = ["long_job", "medium_job", "short_job"]
    else:
        if type(jobs) is not int or jobs < 1:
            raise ValueError("jobs must be a positive integer")
        rng = random.Random(seed)
        durations = [rng.randint(10, 1000) for _ in range(jobs)]
        names = [f"job_{index}" for index in range(jobs)]
    if arrival_times_us is not None:
        if len(arrival_times_us) != len(durations):
            raise ValueError("arrival_times_us length must equal the number of jobs")
        arrivals = list(arrival_times_us)
        arrival_model = "explicit"
    else:
        arrivals = [index * arrival_interval_us for index in range(len(durations))]
        arrival_model = "fixed_interval" if arrival_interval_us else "all_at_zero"
    if any(
        type(value) is not int or not 0 <= value <= MAX_ARRIVAL_US for value in arrivals
    ):
        raise ValueError(
            f"arrival times must be non-negative integer microseconds <= {MAX_ARRIVAL_US}"
        )

    trace = EventTrace(
        num_ranks=1,
        events=[
            Event(
                index,
                EventType.COMPUTE,
                0,
                name,
                duration_us=duration,
                arrival_us=arrivals[index],
            )
            for index, (name, duration) in enumerate(zip(names, durations))
        ],
        parameters={
            "workload": "single_processor_demo",
            "timing_source": "synthetic",
            "job_durations_us": durations,
            "job_arrivals_us": arrivals,
            "arrival_model": arrival_model,
            "scheduling_overhead_us": 0,
            "compute_resources": 1,
        },
    )
    if not any(arrivals):
        trace.parameters["arrival_us"] = 0
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

    Follow the explicit non-preemptive schedule, waiting for each job's release
    even if a later job is ready. Jobs are independent before serialization;
    there is no context-switch overhead. Execution is max(duration_us, 0.001),
    matching ASTRA's minimum runtime of 1 ns for zero-duration compute events.
    Completion and makespan are absolute times; flow is completion minus release.
    This is restricted to the demo, not a general DAG/network timing model.
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
        start = max(elapsed, event.arrival_us)
        elapsed = start + max(event.duration_us, 0.001)
        tasks.append(
            {
                "event_id": event_id,
                "name": event.name,
                "duration_us": event.duration_us,
                "arrival_us": event.arrival_us,
                "start_us": start,
                "completion_us": elapsed,
                "waiting_us": start - event.arrival_us,
                "flow_us": elapsed - event.arrival_us,
                "response_us": start - event.arrival_us,
            }
        )
        placed.add(event_id)
    return {
        "source": "analytical_from_trace_durations_not_astra_timestamps",
        "mean_completion_us": sum(task["completion_us"] for task in tasks) / len(tasks),
        "mean_waiting_us": sum(task["waiting_us"] for task in tasks) / len(tasks),
        "mean_flow_us": sum(task["flow_us"] for task in tasks) / len(tasks),
        "mean_response_us": sum(task["response_us"] for task in tasks) / len(tasks),
        "makespan_us": elapsed,
        "tasks": tasks,
    }
