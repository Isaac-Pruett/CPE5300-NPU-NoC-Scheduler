"""Offline policy integration; does not require Chakra or an ASTRA binary."""

import unittest

from astra.run import apply_schedule
from schedulers.fifo import FIFO
from schedulers.sjf import ShortestJobFirst
from speculative.event_model import Event, EventTrace, EventType
from speculative.speculative_decode import build_speculative_iteration
from workloads.prefill_decode import build_prefill_decode


def compute(event_id, duration, rank=0, dependencies=None):
    return Event(
        event_id,
        EventType.COMPUTE,
        rank,
        f"compute-{event_id}",
        dependencies=dependencies or [],
        duration_us=duration,
    )


class SchedulingTests(unittest.TestCase):
    def test_policies_enforce_different_orders_without_mutating_input(self):
        trace = EventTrace(1, [compute(0, 100), compute(1, 10)])
        before = trace.to_dict()
        fifo = apply_schedule(trace, FIFO)
        sjf = apply_schedule(trace, ShortestJobFirst)
        self.assertEqual(fifo.parameters["compute_schedule"], {"0": [0, 1]})
        self.assertEqual(fifo.events[1].dependencies, [0])
        self.assertEqual(sjf.parameters["compute_schedule"], {"0": [1, 0]})
        self.assertEqual(sjf.events[0].dependencies, [1])
        self.assertEqual(trace.to_dict(), before)

    def test_shorter_job_cannot_bypass_dependency(self):
        trace = EventTrace(1, [compute(0, 100), compute(1, 1, dependencies=[0])])
        scheduled = apply_schedule(trace, ShortestJobFirst)
        self.assertEqual(scheduled.parameters["compute_schedule"], {"0": [0, 1]})
        self.assertEqual(scheduled.events[1].dependencies, [0])

    def test_ranks_are_not_serialized_together(self):
        trace = EventTrace(2, [compute(0, 100), compute(1, 10, rank=1)])
        scheduled = apply_schedule(trace, FIFO)
        self.assertTrue(all(not event.dependencies for event in scheduled.events))
        self.assertEqual(scheduled.parameters["compute_schedule"], {"0": [0], "1": [1]})

    def test_receive_waits_for_matching_send_in_offline_order(self):
        trace = EventTrace(
            2,
            [
                compute(0, 100),
                Event(
                    1,
                    EventType.SEND,
                    0,
                    "send",
                    dependencies=[0],
                    src=0,
                    dst=1,
                    bytes=4,
                    tag="message",
                ),
                Event(
                    2,
                    EventType.RECV,
                    1,
                    "receive",
                    src=0,
                    dst=1,
                    bytes=4,
                    tag="message",
                ),
                compute(3, 1, rank=1, dependencies=[2]),
                compute(4, 100, rank=1),
            ],
        )
        scheduled = apply_schedule(trace, ShortestJobFirst)
        self.assertEqual(scheduled.parameters["compute_schedule"]["1"], [4, 3])
        self.assertEqual(scheduled.events[3].dependencies, [2, 4])
        # Cross-rank SEND edges are traversal-only, never Chakra data dependencies.
        self.assertEqual(scheduled.events[2].dependencies, [])

    def test_communication_deadlock_is_rejected(self):
        trace = EventTrace(
            2,
            [
                Event(
                    0,
                    EventType.SEND,
                    0,
                    "send",
                    dependencies=[1],
                    src=0,
                    dst=1,
                    bytes=4,
                    tag="out",
                ),
                Event(
                    1, EventType.RECV, 0, "receive", src=1, dst=0, bytes=4, tag="back"
                ),
                Event(
                    2,
                    EventType.SEND,
                    1,
                    "send",
                    dependencies=[3],
                    src=1,
                    dst=0,
                    bytes=4,
                    tag="back",
                ),
                Event(
                    3, EventType.RECV, 1, "receive", src=0, dst=1, bytes=4, tag="out"
                ),
            ],
        )
        trace.validate()
        with self.assertRaisesRegex(ValueError, "deadlock"):
            apply_schedule(trace, FIFO)

    def test_existing_workloads_preserve_communication(self):
        for build in (build_prefill_decode, build_speculative_iteration):
            for policy in (FIFO, ShortestJobFirst):
                with self.subTest(workload=build.__name__, policy=policy.__name__):
                    trace = build()
                    scheduled = apply_schedule(trace, policy)
                    scheduled.validate()
                    before = [
                        event.to_dict()
                        for event in trace.events
                        if event.event_type is not EventType.COMPUTE
                    ]
                    after = [
                        event.to_dict()
                        for event in scheduled.events
                        if event.event_type is not EventType.COMPUTE
                    ]
                    self.assertEqual(after, before)
                    expected = {
                        event.event_id
                        for event in trace.events
                        if event.event_type is EventType.COMPUTE
                    }
                    actual = [
                        event_id
                        for ids in scheduled.parameters["compute_schedule"].values()
                        for event_id in ids
                    ]
                    self.assertEqual(set(actual), expected)
                    self.assertEqual(len(actual), len(expected))


if __name__ == "__main__":
    unittest.main()
