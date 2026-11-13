"""Compare completed ASTRA runs and plot simulated completion cycles."""

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
                for event_id in ids
            ]
            print(f"  Rank {rank}: {' -> '.join(names) or '(no compute)'}")
    print(
        "\nExposed communication includes dependency waits; it is not link utilization."
    )
    if len(runs) > 1 and len({run.completion_cycles for run in runs}) == 1:
        print(
            "All runs have identical completion cycles; sequential workloads may offer no policy choice."
        )


def save_chart(runs: list[Run], output: Path, warnings: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
