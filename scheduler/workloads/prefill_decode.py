"""A two-stage inference pipeline with explicit prefill and decode phases.

Ranks represent two pipeline stages, not disaggregated prefill/decode servers.
Both ranks execute each phase. Feedback gates the next autoregressive step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from speculative.event_model import Event, EventTrace, EventType


def build_prefill_decode(
    decode_tokens: int = 4,
    prefill_compute_us: int = 1000,
    decode_compute_us: int = 100,
    prefill_bytes: int = 1048576,
    decode_bytes: int = 4096,
) -> EventTrace:
    """Generate one prompt pass followed by decode_tokens incremental passes.

    Durations are per stage; payloads are synthetic activation sizes. Prefill
    feedback represents the first sampled token, followed by additional decode
    steps. No KV-cache, attention kernels, or router microarchitecture is modeled.
    """
    if decode_tokens < 1:
        raise ValueError("decode_tokens must be positive")
    if min(prefill_compute_us, decode_compute_us) < 0:
        raise ValueError("compute durations must be non-negative")
    if min(prefill_bytes, decode_bytes) < 1:
        raise ValueError("activation byte counts must be positive")

    events: list[Event] = []
    last: dict[int, int] = {}

    def append(kind: EventType, rank: int, name: str, phase: str, **fields) -> None:
        event = Event(
            event_id=len(events), event_type=kind, rank=rank, name=name,
            phase=phase, dependencies=[last[rank]] if rank in last else [],
            **fields,
        )
        events.append(event)
        last[rank] = event.event_id

    for step in range(decode_tokens + 1):
        phase = "prefill" if step == 0 else "decode"
        label = "prefill" if step == 0 else f"decode_{step}"
        duration = prefill_compute_us if step == 0 else decode_compute_us
        size = prefill_bytes if step == 0 else decode_bytes
        append(EventType.COMPUTE, 0, f"{label}_stage_0", phase, duration_us=duration)
        fields = dict(src=0, dst=1, bytes=size, tag=f"{label}_activation")
        append(EventType.SEND, 0, f"{label}_send_activation", phase, **fields)
        append(EventType.RECV, 1, f"{label}_recv_activation", phase, **fields)
        append(EventType.COMPUTE, 1, f"{label}_stage_1", phase, duration_us=duration)
        fields = dict(src=1, dst=0, bytes=4, tag=f"{label}_feedback")
        append(EventType.SEND, 1, f"{label}_send_token", phase, **fields)
        append(EventType.RECV, 0, f"{label}_recv_token", phase, **fields)

    trace = EventTrace(num_ranks=2, events=events, parameters={
        "workload": "prefill_decode_pipeline",
        "timing_source": "synthetic",
        "decode_tokens": decode_tokens,
        "prefill_compute_us": prefill_compute_us,
        "decode_compute_us": decode_compute_us,
        "prefill_bytes": prefill_bytes,
        "decode_bytes": decode_bytes,
    })
    trace.validate()
    return trace


def add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--prefill-compute-us", type=int, default=1000)
    parser.add_argument("--decode-compute-us", type=int, default=100)
    parser.add_argument("--prefill-bytes", type=int, default=1048576)
    parser.add_argument("--decode-bytes", type=int, default=4096)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workload_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("traces/prefill_decode_events.json"))
    args = parser.parse_args()
    trace = build_prefill_decode(**{key: value for key, value in vars(args).items() if key != "output"})
    print(trace.write_json(args.output))


if __name__ == "__main__":
    main()
