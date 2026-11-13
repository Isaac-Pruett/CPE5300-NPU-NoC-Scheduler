"""Synthetic fork/join workload exposing a FIFO versus SJF scheduling choice."""

from speculative.event_model import Event, EventTrace, EventType


def build_policy_demo() -> EventTrace:
    """A short producer unlocks remote work while a long local task can overlap.

    Both rank-0 tasks are initially eligible. FIFO executes the long task first;
    SJF releases rank-1 work earlier. A feedback receive and final join ensure
    rank 0 waits for both branches. Durations are synthetic, not measurements.
    """
    activation = dict(src=0, dst=1, bytes=4, tag="producer_output")
    feedback = dict(src=1, dst=0, bytes=4, tag="consumer_done")
    trace = EventTrace(
        num_ranks=2,
        events=[
            Event(0, EventType.COMPUTE, 0, "long_local", duration_us=1000),
            Event(1, EventType.COMPUTE, 0, "short_producer", duration_us=100),
            Event(2, EventType.SEND, 0, "send_output", dependencies=[1], **activation),
            Event(3, EventType.RECV, 1, "receive_output", **activation),
            Event(
                4,
                EventType.COMPUTE,
                1,
                "remote_consumer",
                dependencies=[3],
                duration_us=1000,
            ),
            Event(5, EventType.SEND, 1, "send_done", dependencies=[4], **feedback),
            Event(6, EventType.RECV, 0, "receive_done", **feedback),
            Event(7, EventType.COMPUTE, 0, "join", dependencies=[0, 6], duration_us=1),
        ],
        parameters={
            "workload": "policy_demo",
            "timing_source": "synthetic",
            "long_local_us": 1000,
            "short_producer_us": 100,
            "remote_consumer_us": 1000,
            "join_us": 1,
            "message_bytes": 4,
        },
    )
    trace.validate()
    return trace
