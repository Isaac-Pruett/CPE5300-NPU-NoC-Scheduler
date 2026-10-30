"""Generate one minimal two-rank speculative-decoding iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

from .event_model import Event, EventTrace, EventType


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "traces" / "speculative_events.json"


def build_speculative_iteration(
    k: int = 4,
    accepted_tokens: int = 3,
    draft_compute_us: int = 100,
    target_verify_us: int = 500,
) -> EventTrace:
    """Build a draft-rank/target-rank speculative-decoding event graph."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 <= accepted_tokens <= k:
        raise ValueError("accepted_tokens must be between 0 and k")
    if draft_compute_us < 0 or target_verify_us < 0:
        raise ValueError("compute durations must be non-negative")

    events: list[Event] = []
    next_event_id = 0
    previous_draft_id: int | None = None

    for token_index in range(1, k + 1):
        event = Event(
            event_id=next_event_id,
            event_type=EventType.COMPUTE,
            rank=0,
            name=f"draft_token_{token_index}",
            duration_us=draft_compute_us,
            dependencies=[] if previous_draft_id is None else [previous_draft_id],
        )
        events.append(event)
        previous_draft_id = event.event_id
        next_event_id += 1

    candidate_send = Event(
        event_id=next_event_id,
        event_type=EventType.SEND,
        rank=0,
        name="send_candidate_tokens",
        src=0,
        dst=1,
        bytes=k * 4,
        tag="candidate_tokens",
        dependencies=[previous_draft_id] if previous_draft_id is not None else [],
    )
    events.append(candidate_send)
    next_event_id += 1

    acceptance_recv = Event(
        event_id=next_event_id,
        event_type=EventType.RECV,
        rank=0,
        name="receive_acceptance_result",
        src=1,
        dst=0,
        bytes=4,
        tag="acceptance_result",
        dependencies=[candidate_send.event_id],
    )
    events.append(acceptance_recv)
    next_event_id += 1

    candidate_recv = Event(
        event_id=next_event_id,
        event_type=EventType.RECV,
        rank=1,
        name="receive_candidate_tokens",
        src=0,
        dst=1,
        bytes=k * 4,
        tag="candidate_tokens",
    )
    events.append(candidate_recv)
    next_event_id += 1

    verification = Event(
        event_id=next_event_id,
        event_type=EventType.COMPUTE,
        rank=1,
        name="target_verification",
        duration_us=target_verify_us,
        dependencies=[candidate_recv.event_id],
    )
    events.append(verification)
    next_event_id += 1

    acceptance_send = Event(
        event_id=next_event_id,
        event_type=EventType.SEND,
        rank=1,
        name="send_acceptance_result",
        src=1,
        dst=0,
        bytes=4,
        tag="acceptance_result",
        dependencies=[verification.event_id],
    )
    events.append(acceptance_send)

    trace = EventTrace(
        num_ranks=2,
        parameters={
            "k": k,
            "accepted_tokens": accepted_tokens,
            "draft_compute_us": draft_compute_us,
            "target_verify_us": target_verify_us,
            "timing_source": "synthetic",
        },
        events=events,
    )
    trace.validate()
    return trace


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate one synthetic two-rank speculative-decoding iteration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k", type=int, default=4, help="Number of draft tokens")
    parser.add_argument(
        "--accepted-tokens",
        type=int,
        default=3,
        help="Number of draft tokens accepted by the target",
    )
    parser.add_argument(
        "--draft-compute-us",
        type=int,
        default=100,
        help="Synthetic compute duration for each draft token",
    )
    parser.add_argument(
        "--target-verify-us",
        type=int,
        default=500,
        help="Synthetic target verification duration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON event trace",
    )
    return parser.parse_args()


def main() -> None:
    """Generate, validate, and write one event trace."""
    args = parse_args()
    trace = build_speculative_iteration(
        k=args.k,
        accepted_tokens=args.accepted_tokens,
        draft_compute_us=args.draft_compute_us,
        target_verify_us=args.target_verify_us,
    )
    output_path = trace.write_json(args.output)

    print("Speculative-decoding event trace generated")
    print(f"  ranks:  {trace.num_ranks}")
    print(f"  events: {len(trace.events)}")
    print(f"  output: {output_path}")
    print("  timing: synthetic")


if __name__ == "__main__":
    main()
