"""Generate a workload, convert it, and run ASTRA-sim in one container."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from astra.configuration import available_configs, load_config
from schedulers.discovery import discover_schedulers
from schedulers.scheduler import Scheduler
from schedulers.task import Task
from speculative.event_model import EventTrace, EventType
from speculative.speculative_decode import build_speculative_iteration
from workloads.policy_demo import build_policy_demo
from workloads.prefill_decode import add_workload_arguments, build_prefill_decode
from workloads.single_processor import (
    build_single_processor_demo,
    single_processor_metrics,
)

BINARIES = {
    "congestion-aware": "AstraSim_Analytical_Congestion_Aware",
    "congestion-unaware": "AstraSim_Analytical_Congestion_Unaware",
}


def apply_schedule(
    trace: EventTrace,
    policy_factory: Callable[[], Scheduler],
) -> EventTrace:
    """Return an offline, non-preemptive compute schedule, one resource per rank.

    Traversal means placement in the schedule, not simulated completion. A fresh
    policy selects from each ready-set snapshot so policies that requeue unfinished
    tasks can be used without pretending those tasks have executed.
    """
    scheduled = EventTrace.from_dict(trace.to_dict())
    by_id = {event.event_id: event for event in scheduled.events}
    dependencies = {
        event.event_id: set(event.dependencies) for event in scheduled.events
    }
    sends = {}
    for event in scheduled.events:
        if event.event_type is EventType.SEND:
            key = (event.src, event.dst, event.tag, event.bytes)
            if key in sends:
                raise ValueError(
                    "offline scheduling requires unique communication tags per pair"
                )
            sends[key] = event.event_id
    for event in scheduled.events:
        if event.event_type is EventType.RECV:
            dependencies[event.event_id].add(
                sends[(event.src, event.dst, event.tag, event.bytes)]
            )

    pending = set(by_id)
    placed: set[int] = set()
    previous_compute: dict[int, int] = {}
    order: dict[str, list[int]] = {str(rank): [] for rank in range(trace.num_ranks)}
    while pending:
        # Communication advances the symbolic DAG without occupying compute.
        ready_communication = [
            event_id
            for event_id in sorted(pending)
            if by_id[event_id].event_type is not EventType.COMPUTE
            and dependencies[event_id] <= placed
        ]
        if ready_communication:
            pending.difference_update(ready_communication)
            placed.update(ready_communication)
            continue

        progressed = False
        for rank in range(trace.num_ranks):
            ready = [
                by_id[event_id]
                for event_id in sorted(pending)
                if by_id[event_id].event_type is EventType.COMPUTE
                and by_id[event_id].rank == rank
                and dependencies[event_id] <= placed
            ]
            if not ready:
                continue
            policy = policy_factory()
            tasks = [Task(event=event) for event in ready]
            policy.admit(tasks)
            selected = policy.next()
            if not any(selected is task for task in tasks):
                raise ValueError(
                    "scheduler must return one of its admitted ready tasks"
                )
            event = selected.event
            previous = previous_compute.get(rank)
            if previous is not None and previous not in event.dependencies:
                event.dependencies.append(previous)
            previous_compute[rank] = event.event_id
            order[str(rank)].append(event.event_id)
            pending.remove(event.event_id)
            placed.add(event.event_id)
            progressed = True
        if not progressed:
            raise ValueError(
                f"communication/dependency deadlock in offline schedule; blocked events: {sorted(pending)}"
            )

    scheduled.parameters["compute_schedule"] = order
    scheduled.validate()
    return scheduled


def completed_ranks(output: str, num_ranks: int) -> dict[str, dict[str, int]]:
    """Require explicit completion from every rank, not just an exit code of 0."""
    ranks = {}
    for rank, cycles, exposed in re.findall(
        r"sys\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles",
        output,
    ):
        ranks[rank] = {
            "cycles": int(cycles),
            "exposed_communication_cycles": int(exposed),
        }
    if set(ranks) != {str(rank) for rank in range(num_ranks)}:
        raise RuntimeError(
            "ASTRA-sim did not report completion for every rank; inspect simulator.log"
        )
    for rank, cycles in re.findall(r"sys\[(\d+)\], GPU time: (\d+)", output):
        if rank in ranks:
            ranks[rank]["compute_cycles"] = int(cycles)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        policies = discover_schedulers()
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument(
        "--workload",
        choices=[
            "prefill-decode",
            "speculative",
            "policy-demo",
            "single-processor-demo",
        ],
        default="prefill-decode",
    )
    parser.add_argument("--backend", choices=BINARIES, default="congestion-aware")
    parser.add_argument(
        "--config-dir",
        default="two_rank",
        help="Bundled hardware preset name or directory path (default: two_rank)",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List bundled hardware presets and exit",
    )
    parser.add_argument(
        "--policy",
        choices=["none", *policies],
        default="fifo",
        help="Offline non-preemptive compute ordering per rank; none preserves the original graph",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List automatically discovered schedulers and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory; existing runs are never overwritten",
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Simulator timeout in seconds"
    )
    add_workload_arguments(parser)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--accepted-tokens", type=int, default=3)
    parser.add_argument("--draft-compute-us", type=int, default=100)
    parser.add_argument("--target-verify-us", type=int, default=500)
    parser.add_argument(
        "--jobs",
        type=int,
        help="Generated single-processor job count; omitted keeps the fixed three-job demo",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for generated single-processor jobs"
    )
    args = parser.parse_args()
    if args.list_policies:
        print("none: preserve original workload dependencies")
        for name, policy in policies.items():
            print(f"{name}: {policy.__module__}.{policy.__name__}")
        return
    if args.list_configs:
        for directory in available_configs():
            try:
                config = load_config(directory, backend=args.backend)
            except ValueError as exc:
                parser.error(str(exc))
            print(f"{directory.name}: {config.num_ranks} ranks — {config.description}")
        return
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.jobs is not None:
        if args.workload != "single-processor-demo":
            parser.error("--jobs is only supported for single-processor-demo")
        if args.jobs < 1:
            parser.error("--jobs must be positive")

    if args.workload == "prefill-decode":
        trace = build_prefill_decode(
            args.decode_tokens,
            args.prefill_compute_us,
            args.decode_compute_us,
            args.prefill_bytes,
            args.decode_bytes,
        )
    elif args.workload == "single-processor-demo":
        if args.policy == "none":
            parser.error(
                "single-processor-demo requires an explicit scheduling policy (not none)"
            )
        trace = build_single_processor_demo(jobs=args.jobs, seed=args.seed)
    elif args.workload == "policy-demo":
        trace = build_policy_demo()
    else:
        trace = build_speculative_iteration(
            args.k,
            args.accepted_tokens,
            args.draft_compute_us,
            args.target_verify_us,
        )
    if args.policy != "none":
        trace = apply_schedule(trace, policies[args.policy])
    trace.parameters["scheduling_policy"] = args.policy
    try:
        config = load_config(
            args.config_dir, num_ranks=trace.num_ranks, backend=args.backend
        )
    except ValueError as exc:
        parser.error(str(exc))
    binary = shutil.which(BINARIES[args.backend])
    if binary is None:
        parser.error(
            "ASTRA-sim binary missing. Run this command through docker compose."
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = (
        args.output_dir
        or Path("results/raw") / f"{stamp}-{args.workload}-{args.backend}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace.write_json(output_dir / "events.json")
    from tracegen.spec_to_chakra import convert_trace

    convert_trace(trace, output_dir / "workload")
    config.write_to(output_dir)

    command = [
        binary,
        f"--workload-configuration={output_dir / 'workload'}",
        f"--system-configuration={output_dir / 'system.json'}",
        f"--network-configuration={output_dir / 'network.yml'}",
        f"--remote-memory-configuration={output_dir / 'remote_memory.json'}",
    ]
    revision_file = Path("/opt/astra-sim-revision")
    summary = {
        "status": "running",
        "workload": args.workload,
        "backend": args.backend,
        "timing_source": "synthetic",
        "parameters": trace.parameters,
        "scheduling_policy": args.policy,
        "hardware": config.summary(),
        "astra_sim_revision": revision_file.read_text().strip()
        if revision_file.exists()
        else "unknown",
        "command": command,
    }
    if args.workload == "single-processor-demo":
        summary["single_processor_metrics"] = single_processor_metrics(trace)
    summary_file = output_dir / "summary.json"

    def save_summary() -> None:
        summary_file.write_text(json.dumps(summary, indent=2) + "\n")

    save_summary()
    log = output_dir / "simulator.log"
    print(
        f"Running {args.workload} with {args.backend}, config {config.directory.name}; artifacts: {output_dir}",
        flush=True,
    )
    try:
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=output_dir,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        output = log.read_text()
        print(output, end="")
        summary["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                f"ASTRA-sim exited with {result.returncode}; inspect {log}"
            )
        summary["ranks"] = completed_ranks(output, trace.num_ranks)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        save_summary()
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
