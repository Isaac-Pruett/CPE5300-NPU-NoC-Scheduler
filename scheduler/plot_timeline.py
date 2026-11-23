"""Plot actual ASTRA event_timing.csv callback intervals, never inferred schedules.

ready is the first observation that dependencies (including policy edges) are
satisfied. start/end are the producer's issue/completion callback timestamps.
In particular, SEND spans issue to its recorded completion callback, NOT an
assumed network delivery or matching receive completion. No analytical duration,
queue ordering, or cross-rank timestamp offset is reconstructed here.
"""

import argparse
import csv
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

FIELDS = (
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
)
INTEGER_FIELDS = {"rank", "event_id", "ready_cycle", "start_cycle", "end_cycle"}
NUMERIC_FIELDS = INTEGER_FIELDS | {
    "clock_period_ns",
    "ready_us",
    "start_us",
    "end_us",
    "queue_wait_us",
    "execution_us",
}
COLORS = {"COMPUTE": "#4c78a8", "SEND": "#f58518", "RECV": "#54a24b"}
HATCHES = {"": "", "prefill": "///", "decode": "..."}


@dataclass(frozen=True)
class Event:
    run: int
    rank: int
    name: str
    event_type: str
    phase: str
    start: float
    end: float


def _events(directory: Path, run: int):
    path = directory / "event_timing.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            try:
                if None in row or any(row.get(field) is None for field in FIELDS):
                    raise ValueError("incorrect number of columns")
                values = {}
                for field in NUMERIC_FIELDS:
                    value = (
                        int(row[field])
                        if field in INTEGER_FIELDS
                        else float(row[field])
                    )
                    if value < 0 or (
                        field not in INTEGER_FIELDS and not math.isfinite(value)
                    ):
                        raise ValueError(f"{field} must be finite and nonnegative")
                    values[field] = value
                for suffix in ("cycle", "us"):
                    if (
                        not values[f"ready_{suffix}"]
                        <= values[f"start_{suffix}"]
                        <= values[f"end_{suffix}"]
                    ):
                        raise ValueError(
                            f"expected ready_{suffix} <= start_{suffix} <= end_{suffix}"
                        )
                if row["event_type"] not in COLORS:
                    raise ValueError(f"unknown event_type {row['event_type']!r}")
                if row["phase"] not in HATCHES:
                    raise ValueError(f"unknown phase {row['phase']!r}")
                yield Event(
                    run,
                    int(values["rank"]),
                    row["name"],
                    row["event_type"],
                    row["phase"],
                    values["start_us"],
                    values["end_us"],
                )
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{path}: row {reader.line_num}: {exc}") from exc


def _label(directory: Path) -> str:
    try:
        summary = json.loads((directory / "summary.json").read_text())
    except (OSError, ValueError):
        return directory.name
    policy = summary.get("scheduling_policy") if isinstance(summary, dict) else None
    return f"{directory.name} ({policy})" if policy else directory.name


def save_gantt(
    run_dirs: list[Path],
    output: Path,
    *,
    start_us: float | None = None,
    end_us: float | None = None,
    max_events: int = 500,
) -> None:
    """Save a globally aligned timeline from one or more run directories.

    Window bounds are inclusive: events intersecting the window are retained,
    with bars clipped by the axes rather than changing their recorded intervals.
    The earliest max_events matching operations (stable CSV order for ties) are
    displayed across all runs. All input rows are validated even when excluded;
    selection uses O(max_events) storage. Missing summaries are supported.
    """
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    if (
        isinstance(max_events, bool)
        or not isinstance(max_events, int)
        or max_events <= 0
    ):
        raise ValueError("max_events must be a positive integer")
    for name, value in (("start_us", start_us), ("end_us", end_us)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if start_us is not None and end_us is not None and start_us >= end_us:
        raise ValueError("start_us must be less than end_us")

    selected = []
    total = matched = 0
    first, last = math.inf, 0.0
    for run, directory in enumerate(run_dirs):
        for event in _events(directory, run):
            total += 1
            if (start_us is not None and event.end < start_us) or (
                end_us is not None and event.start > end_us
            ):
                continue
            matched += 1
            first, last = min(first, event.start), max(last, event.end)
            entry = (-event.start, -total, event)
            if len(selected) < max_events:
                heapq.heappush(selected, entry)
            elif entry[:2] > selected[0][:2]:
                heapq.heapreplace(selected, entry)
    if not matched:
        raise ValueError(
            "no events intersect the requested time window (or CSV is empty)"
        )
    events = [
        entry[2] for entry in sorted(selected, key=lambda item: (-item[0], -item[1]))
    ]

    # Pack overlapping operations into sublanes, independently for each rank/type.
    groups = {}
    for event in events:
        groups.setdefault((event.run, event.rank, event.event_type), []).append(event)
    bars, ticks, labels = [], [], []
    lane = 0
    for (run, rank, kind), operations in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], list(COLORS).index(item[0][2])),
    ):
        lane_ends = []
        for event in operations:
            sublane = next(
                (i for i, end in enumerate(lane_ends) if end <= event.start),
                len(lane_ends),
            )
            if sublane == len(lane_ends):
                lane_ends.append(event.end)
            else:
                lane_ends[sublane] = event.end
            bars.append((lane + sublane, event))
        ticks.append(lane + (len(lane_ends) - 1) / 2)
        labels.append(f"{_label(run_dirs[run])} / rank {rank} / {kind}")
        lane += len(lane_ends)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(14, max(4, min(40, lane * 0.35 + 2))))
    try:
        for y, event in bars:
            ax.barh(
                y,
                event.end - event.start,
                left=event.start,
                height=0.75,
                color=COLORS[event.event_type],
                hatch=HATCHES[event.phase],
                edgecolor="black",
                linewidth=0.4,
            )
            if event.end == event.start:
                ax.plot(event.start, y, marker="|", color=COLORS[event.event_type])
            if len(events) <= 40:
                ax.text(
                    event.start, y, event.name, va="center", fontsize=7, clip_on=True
                )
        ax.set_yticks(ticks, labels)
        ax.invert_yaxis()
        left = start_us if start_us is not None else min(0.0, first)
        right = end_us if end_us is not None else last
        if left == right:
            right = left + 1.0
        ax.set_xlim(left, right)
        ax.set_xlabel("Simulated time (µs; shared origin across ranks)")
        title = f"ASTRA callback timeline — displayed {len(events)} / {matched} events in window ({total} total)"
        if len(events) < matched:
            title += f"\nTRUNCATED: earliest {max_events} matching events only"
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        handles = [Patch(facecolor=color, label=kind) for kind, color in COLORS.items()]
        handles += [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=hatch,
                label=phase or "unspecified phase",
            )
            for phase, hatch in HATCHES.items()
        ]
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6
        )
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/gantt.png"))
    parser.add_argument("--start-us", type=float)
    parser.add_argument("--end-us", type=float)
    parser.add_argument("--max-events", type=int, default=500)
    args = parser.parse_args()
    try:
        save_gantt(
            args.run_dirs,
            args.output,
            start_us=args.start_us,
            end_us=args.end_us,
            max_events=args.max_events,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
