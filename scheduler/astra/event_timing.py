"""Validate and export simulator-observed event lifecycle timestamps."""

import csv
import json
import re
from pathlib import Path

from speculative.event_model import EventTrace

PREFIX = "ASTRA_EVENT_TIMING_V1,"
RECORD = re.compile(r"ASTRA_EVENT_TIMING_V1,(\d+),(\d+),(ready|start|end),(\d+),(\d+)")
FIELDS = [
    "rank",
    "event_id",
    "name",
    "event_type",
    "phase",
    "ready_cycle",
    "start_cycle",
    "end_cycle",
    "clock_period_ns",
    "ready_us",
    "start_us",
    "end_us",
    "queue_wait_us",
    "execution_us",
    "arrival_us",
    "arrival_cycle",
]


def parse_event_timings(output: str, trace: EventTrace, ranks: dict) -> list[dict]:
    """Require exactly one ready/start/end observation for every trace event.

    Ready is first observed released and dependency-free in ASTRA, including
    policy-inserted edges. SEND end means its ASTRA callback, not delivery.
    Arrivals are declared workload releases, never observed callbacks.
    No durations or missing callback timestamps are inferred from the workload.
    """
    expected = {(event.rank, event.event_id): event for event in trace.events}
    observations = {}
    clock_period = None
    for line in output.splitlines():
        if "ASTRA_EVENT_TIMING" not in line:
            continue
        match = RECORD.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"malformed ASTRA timing record: {line}")
        rank, event_id, stage, cycle, period = match.groups()
        key = (int(rank), int(event_id))
        cycle, period = int(cycle), int(period)
        if key not in expected:
            raise ValueError(f"unexpected ASTRA timing event: {key}")
        if period <= 0 or (clock_period is not None and period != clock_period):
            raise ValueError(
                "ASTRA timing records have invalid or inconsistent clock periods"
            )
        clock_period = period
        stages = observations.setdefault(key, {})
        if stage in stages:
            raise ValueError(f"duplicate ASTRA timing {stage} for event {key}")
        stages[stage] = cycle
    if not observations:
        raise ValueError(
            "ASTRA event timestamps missing; rebuild with 'docker compose build' for event-timing-v1"
        )
    rows = []
    for key, event in expected.items():
        stages = observations.get(key, {})
        if set(stages) != {"ready", "start", "end"}:
            raise ValueError(f"incomplete ASTRA timing records for event {key}")
        ready, start, end = (stages[name] for name in ("ready", "start", "end"))
        arrival_cycle, remainder = divmod(event.arrival_us * 1000, clock_period)
        if remainder:
            raise ValueError(f"event {key} arrival is not an exact clock cycle")
        if ready < arrival_cycle or start < arrival_cycle:
            raise ValueError(f"ASTRA event {key} is ready or starts before its arrival")
        if not ready <= start <= end:
            raise ValueError(f"non-monotonic ASTRA timing records for event {key}")
        if str(event.rank) not in ranks or end > ranks[str(event.rank)]["cycles"]:
            raise ValueError(f"event {key} ends after its rank completion")
        factor = clock_period / 1000
        rows.append(
            {
                "rank": event.rank,
                "event_id": event.event_id,
                "name": event.name,
                "event_type": event.event_type.value,
                "phase": event.phase or "",
                "arrival_us": event.arrival_us,
                "arrival_cycle": arrival_cycle,
                "ready_cycle": ready,
                "start_cycle": start,
                "end_cycle": end,
                "clock_period_ns": clock_period,
                "ready_us": ready * factor,
                "start_us": start * factor,
                "end_us": end * factor,
                "queue_wait_us": (start - ready) * factor,
                "execution_us": (end - start) * factor,
            }
        )
    by_id = {row["event_id"]: row for row in rows}
    for event in trace.events:
        row = by_id[event.event_id]
        if any(
            by_id[dependency]["end_cycle"] > row["ready_cycle"]
            for dependency in event.dependencies
        ):
            raise ValueError(
                f"ASTRA event {event.event_id} is ready before a dependency completes"
            )
    for rank in range(trace.num_ranks):
        compute = sorted(
            (
                row
                for row in rows
                if row["rank"] == rank and row["event_type"] == "COMPUTE"
            ),
            key=lambda row: (row["start_cycle"], row["end_cycle"]),
        )
        if any(
            previous["end_cycle"] > current["start_cycle"]
            for previous, current in zip(compute, compute[1:])
        ):
            raise ValueError(
                f"overlapping ASTRA compute events on the single rank-{rank} resource"
            )
    return sorted(
        rows, key=lambda row: (row["start_cycle"], row["rank"], row["event_id"])
    )


def write_event_timings(
    output: str, trace: EventTrace, ranks: dict, directory: Path
) -> tuple[dict, list[dict]]:
    rows = parse_event_timings(output, trace, ranks)
    with (directory / "event_timing.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    transitions = {}
    for row in rows:
        reference = {"rank": row["rank"], "event_id": row["event_id"]}
        for stage, field in (
            ("arrived", "arrival_cycle"),
            ("ready", "ready_cycle"),
            ("started", "start_cycle"),
            ("completed", "end_cycle"),
        ):
            cycle = row[field]
            step = transitions.setdefault(
                cycle,
                {
                    "cycle": cycle,
                    "time_us": cycle * row["clock_period_ns"] / 1000,
                    "arrived": [],
                    "ready": [],
                    "started": [],
                    "completed": [],
                },
            )
            step[stage].append(reference)
    timing = {
        "source": "astra_sim_callbacks",
        "schema_version": 2,
        "arrival_source": "workload_release_times",
        "clock_period_ns": rows[0]["clock_period_ns"],
        "event_count": len(rows),
        "csv": "event_timing.csv",
        "timesteps": "timesteps.json",
        "gantt": "gantt.png",
        "ready_semantics": "first observed released and dependency-free, including policy-inserted edges",
        "end_semantics": "ASTRA completion callback; SEND completion is not necessarily delivery",
    }
    (directory / "timesteps.json").write_text(
        json.dumps(
            {
                "source": timing["source"],
                "schema_version": 2,
                "arrival_source": timing["arrival_source"],
                "clock_period_ns": timing["clock_period_ns"],
                "steps": [transitions[cycle] for cycle in sorted(transitions)],
            },
            indent=2,
        )
        + "\n"
    )
    return timing, rows


def simulated_single_processor_metrics(rows: list[dict], trace: EventTrace) -> dict:
    """Callback execution times relative to declared workload releases."""
    if (
        trace.num_ranks != 1
        or trace.parameters.get("workload") != "single_processor_demo"
    ):
        raise ValueError("simulated latency requires the single-processor demo")
    if not rows or any(
        row["rank"] != 0 or row["event_type"] != "COMPUTE" for row in rows
    ):
        raise ValueError(
            "simulated single-processor metrics require rank-0 compute events"
        )
    by_id = {event.event_id: event for event in trace.events}
    tasks = [
        {
            "event_id": row["event_id"],
            "name": row["name"],
            "arrival_us": by_id[row["event_id"]].arrival_us,
            "duration_us": row["execution_us"],
            "start_us": row["start_us"],
            "completion_us": row["end_us"],
            "flow_us": row["end_us"] - by_id[row["event_id"]].arrival_us,
            "response_us": row["start_us"] - by_id[row["event_id"]].arrival_us,
            "waiting_us": row["start_us"] - by_id[row["event_id"]].arrival_us,
        }
        for row in rows
    ]
    return {
        "source": "astra_sim_callbacks",
        "arrival_source": "workload_release_times",
        "arrival_assumption": "declared workload releases (arrival_us), not observed callbacks",
        "mean_flow_us": sum(task["flow_us"] for task in tasks) / len(tasks),
        "mean_response_us": sum(task["response_us"] for task in tasks) / len(tasks),
        "mean_completion_us": sum(task["completion_us"] for task in tasks) / len(tasks),
        "mean_waiting_us": sum(task["waiting_us"] for task in tasks) / len(tasks),
        "makespan_us": max(task["completion_us"] for task in tasks),
        "tasks": tasks,
    }
