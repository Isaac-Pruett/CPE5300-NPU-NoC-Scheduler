"""Compare ASTRA completion cycles, callback latency, and analytical references."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILES = ("system.json", "network.yml", "remote_memory.json")
SCHEDULING_FIELDS = {"compute_schedule", "scheduling_policy"}


@dataclass
class Run:
    directory: Path
    summary: dict
    events: dict

    @property
    def label(self) -> str:
        policy = self.summary.get("scheduling_policy", "unknown")
        return f"{self.directory.name} ({policy})"

    @property
    def completion_cycles(self) -> int:
        return max(rank["cycles"] for rank in self.summary["ranks"].values())


def load_run(path: Path) -> Run:
    directory = path.parent if path.name == "summary.json" else path
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "completed":
        raise ValueError(
            f"{directory}: run is not completed (status={summary.get('status')!r})"
        )
    events = json.loads((directory / "events.json").read_text())
    num_ranks = events.get("num_ranks")
    if type(num_ranks) is not int or num_ranks <= 0:
        raise ValueError(f"{directory}: invalid num_ranks in events.json")
    ranks = summary.get("ranks", {})
    if set(ranks) != {str(rank) for rank in range(num_ranks)}:
        raise ValueError(f"{directory}: missing or unexpected rank completion records")
    for rank, metrics in ranks.items():
        for field in ("cycles", "compute_cycles", "exposed_communication_cycles"):
            value = metrics.get(field)
            if (field == "cycles" or value is not None) and (
                type(value) is not int or value < 0
            ):
                raise ValueError(f"{directory}: rank {rank} has invalid {field}")
    return Run(directory, summary, events)


def comparison_warnings(runs: list[Run]) -> list[str]:
    """Flag known confounders; matching metadata is not proof of equivalence."""
    warnings = []
    baseline = runs[0]
    for field in ("workload", "backend", "timing_source", "astra_sim_revision"):
        values = [run.summary.get(field) for run in runs]
        if any(value is None or value == "unknown" for value in values):
            warnings.append(
                f"Missing/unknown {field}; comparability cannot be verified."
            )
        elif any(value != values[0] for value in values[1:]):
            warnings.append(f"Runs differ in {field}.")
    parameters = [
        {
            key: value
            for key, value in run.summary.get("parameters", {}).items()
            if key not in SCHEDULING_FIELDS
        }
        for run in runs
    ]
    if any(value != parameters[0] for value in parameters[1:]):
        warnings.append("Runs differ in workload parameters.")
    if any(run.events["num_ranks"] != baseline.events["num_ranks"] for run in runs[1:]):
        warnings.append("Runs differ in rank count.")
    for name in CONFIG_FILES:
        paths = [run.directory / name for run in runs]
        if not all(path.is_file() for path in paths):
            warnings.append(
                f"Missing {name}; configuration equality cannot be verified."
            )
        else:
            contents = [path.read_bytes() for path in paths]
            if any(content != contents[0] for content in contents[1:]):
                warnings.append(f"Runs differ in {name}.")
    return warnings


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    for row in [headers, ["-" * width for width in widths], *rows]:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def latency_field(metrics: dict) -> str:
    """Never treat absolute completion as release-relative flow implicitly."""
    return "mean_flow_us" if "mean_flow_us" in metrics else "mean_completion_us"


def latency_heading(runs: list[Run], key: str) -> str:
    fields = {latency_field(run.summary[key]) for run in runs if key in run.summary}
    if fields == {"mean_flow_us"}:
        return "Mean flow (us)"
    if fields == {"mean_completion_us"}:
        return "Mean completion (us)"
    return "Mean flow / legacy completion (us)"


def latency_basis(metrics: dict) -> str:
    if latency_field(metrics) == "mean_flow_us":
        return "flow = completion - workload release"
    return "legacy absolute completion; assumes all jobs arriving at time 0"


def print_report(runs: list[Run], warnings: list[str]) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")
    baseline = runs[0].completion_cycles
    print("\nCompletion = maximum rank completion cycles; first run is the baseline.")
    rows = []
    for run in runs:
        speedup = (
            f"{baseline / run.completion_cycles:.3f}x"
            if not warnings and run.completion_cycles > 0
            else "N/A"
        )
        rows.append([run.label, f"{run.completion_cycles:,}", speedup])
    print_table(["Run / policy", "Completion cycles", "Speedup vs first"], rows)
    print("\nPer-rank simulated timing (cycles):")
    rows = []
    for run in runs:
        for rank, metrics in sorted(
            run.summary["ranks"].items(), key=lambda item: int(item[0])
        ):
            rows.append(
                [
                    run.label,
                    rank,
                    f"{metrics['cycles']:,}",
                    str(metrics.get("compute_cycles", "N/A")),
                    str(metrics.get("exposed_communication_cycles", "N/A")),
                ]
            )
    print_table(
        ["Run / policy", "Rank", "Total", "Compute", "Exposed communication"], rows
    )
    simulated_rows = []
    task_rows = []
    total_simulated_tasks = 0
    release_tasks = any(
        "mean_flow_us" in run.summary.get("simulated_single_processor_metrics", {})
        for run in runs
    )
    task_fields = ["duration_us", "start_us", "completion_us", "waiting_us"]
    task_headers = ["Duration (us)", "Start (us)", "Completion (us)", "Waiting (us)"]
    if release_tasks:
        task_fields = ["arrival_us", *task_fields, "flow_us", "response_us"]
        task_headers = ["Arrival (us)", *task_headers, "Flow (us)", "Response (us)"]
    for run in runs:
        metrics = run.summary.get("simulated_single_processor_metrics")
        if metrics is None:
            continue
        simulated_rows.append(
            [
                run.label,
                *[
                    f"{metrics[field]:,.3f}"
                    for field in (
                        latency_field(metrics),
                        "mean_waiting_us",
                        "makespan_us",
                    )
                ],
                latency_basis(metrics),
            ]
        )
        total_simulated_tasks += len(metrics["tasks"])
        for task in metrics["tasks"][:25]:
            task_rows.append(
                [
                    run.label,
                    str(task["event_id"]),
                    task["name"],
                    *[
                        f"{task[field]:,.3f}" if field in task else "N/A"
                        for field in task_fields
                    ],
                ]
            )
    if simulated_rows:
        print(
            "\nSIMULATED single-processor latency — actual ASTRA callback measurements:"
        )
        print(
            "Source: astra_sim_callbacks; ASTRA ticks converted from ns to us "
            "(not an analytical estimate). Arrivals are declared workload releases, "
            "not observed callbacks. Flow is release-relative; completion and makespan "
            "are absolute timestamps/horizons. Legacy mean completion assumes "
            "all jobs arriving at time 0; it is not mixed with flow in charts."
        )
        print("Only runs with simulated metrics are shown in this section.")
        print_table(
            [
                "Run / policy",
                latency_heading(runs, "simulated_single_processor_metrics"),
                "Mean waiting (us)",
                "Makespan (us)",
                "Latency basis",
            ],
            simulated_rows,
        )
        print("\nSIMULATED per-job latency (us):")
        print_table(
            [
                "Run / policy",
                "Event ID",
                "Job",
                *task_headers,
            ],
            task_rows,
        )
        if total_simulated_tasks > len(task_rows):
            print(
                "Showing at most 25 jobs per run; full simulated timestamps are in event_timing.csv and summary.json."
            )
    latency_rows = []
    for run in runs:
        metrics = run.summary.get("single_processor_metrics")
        if metrics is not None:
            latency_rows.append(
                [
                    run.label,
                    f"{metrics[latency_field(metrics)]:,.3f}",
                    f"{metrics['mean_waiting_us']:,.3f}",
                    f"{metrics['makespan_us']:,.3f}",
                    latency_basis(metrics),
                ]
            )
    if latency_rows:
        print(
            "\nANALYTICAL single-processor latency reference from trace durations "
            "(not ASTRA timestamps or measurements):"
        )
        print(
            "Declared workload releases for flow; legacy reference assumes "
            "Independent COMPUTE jobs, all arriving at time 0; "
            "one rank, serial schedule, zero overhead. Completion and makespan are absolute."
        )
        print_table(
            [
                "Run / policy",
                latency_heading(runs, "single_processor_metrics"),
                "Mean waiting (us)",
                "Makespan (us)",
                "Latency basis",
            ],
            latency_rows,
        )
    print("\nOffline compute orders (not measured event start/completion times):")
    for run in runs:
        print(f"\n{run.label}")
        by_id = {event["event_id"]: event for event in run.events["events"]}
        schedule = run.summary.get("parameters", {}).get("compute_schedule")
        if not schedule:
            print(
                "  No recorded compute schedule (e.g. --policy none); file order is not execution order."
            )
            continue
        for rank, ids in sorted(schedule.items(), key=lambda item: int(item[0])):
            names = [
                f"{event_id}:{by_id.get(event_id, {}).get('name', '?')}"
                for event_id in ids[:25]
            ]
            suffix = (
                f" ... ({len(ids) - 25} more; full order in summary.json parameters.compute_schedule)"
                if len(ids) > 25
                else ""
            )
            print(f"  Rank {rank}: {' -> '.join(names) or '(no compute)'}{suffix}")
    print(
        "\nExposed communication includes dependency waits; it is not link utilization."
    )
    if len(runs) > 1 and len({run.completion_cycles for run in runs}) == 1:
        if simulated_rows or latency_rows:
            print(
                "Total completion cycles are identical; reordering fixed work on one busy "
                "resource can improve mean job latency without changing total runtime."
            )
            if simulated_rows:
                print(
                    "SIMULATED latency above uses actual ASTRA callbacks, not estimates."
                )
            if latency_rows:
                print(
                    "ANALYTICAL latency above is a separate trace-duration reference."
                )
        else:
            print(
                "All runs have identical completion cycles; sequential workloads may offer no policy choice."
            )


def save_chart(runs: list[Run], output: Path, warnings: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    simulated_present = [
        run.summary.get("simulated_single_processor_metrics") is not None
        for run in runs
    ]
    latency_key = None
    if all(simulated_present):
        latency_key = "simulated_single_processor_metrics"
    elif not any(simulated_present) and all(
        run.summary.get("single_processor_metrics") is not None for run in runs
    ):
        latency_key = "single_processor_metrics"
    if latency_key is not None:
        fields = {latency_field(run.summary[latency_key]) for run in runs}
        if len(fields) != 1:
            # Do not compare release-relative latency with absolute timestamps.
            latency_key = None
    has_latency = latency_key is not None
    if has_latency:
        figure, (axis, latency_axis) = plt.subplots(
            2, 1, figsize=(max(7, len(runs) * 2), 10)
        )
    else:
        figure, axis = plt.subplots(figsize=(max(7, len(runs) * 2), 5))
    positions = list(range(len(runs)))
    bars = axis.bar(positions, [run.completion_cycles for run in runs])
    axis.set_xticks(positions, [run.label for run in runs], rotation=15, ha="right")
    axis.set_ylabel("Simulated completion cycles (max across ranks)")
    axis.set_title("ASTRA-sim policy comparison")
    axis.bar_label(
        bars, labels=[f"{run.completion_cycles:,}" for run in runs], padding=3
    )
    axis.margins(y=0.15)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    if has_latency:
        field = latency_field(runs[0].summary[latency_key])
        is_flow = field == "mean_flow_us"
        metric_name = "flow" if is_flow else "completion"
        arrival_note = (
            "Declared workload releases, not observed arrival callbacks; flow = end - arrival"
            if is_flow
            else "Legacy absolute completion; assumes all jobs arriving at time 0"
        )
        completions = [run.summary[latency_key][field] for run in runs]
        latency_bars = latency_axis.bar(positions, completions)
        latency_axis.set_xticks(
            positions, [run.label for run in runs], rotation=15, ha="right"
        )
        if latency_key == "simulated_single_processor_metrics":
            latency_axis.set_ylabel(f"Simulated mean {metric_name} (us)")
            latency_axis.set_title(
                "SIMULATED actual ASTRA callbacks (astra_sim_callbacks) — ns to us\n"
                + arrival_note
            )
        else:
            latency_axis.set_ylabel(f"Analytical mean {metric_name} (us)")
            latency_axis.set_title(
                "ANALYTICAL from trace durations — not ASTRA measurements\n"
                + arrival_note
            )
        latency_axis.bar_label(
            latency_bars, labels=[f"{value:,.3f}" for value in completions], padding=3
        )
        latency_axis.margins(y=0.15)
        latency_axis.grid(axis="y", alpha=0.25)
        latency_axis.set_axisbelow(True)
    if warnings:
        figure.text(
            0.5,
            0.01,
            "WARNING: runs are not verified comparable; see console warnings.",
            ha="center",
            fontsize=9,
        )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories or summary.json paths; first is baseline",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/policy_comparison.png")
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print tables without requiring matplotlib",
    )
    args = parser.parse_args()
    try:
        runs = [load_run(path) for path in args.runs]
        warnings = comparison_warnings(runs)
        print_report(runs, warnings)
        if not args.no_plot:
            save_chart(runs, args.output, warnings)
            print(f"\nSaved chart: {args.output}")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
