"""Contracts for event ordering, Chakra encoding, and simulator completion."""

import tempfile
from pathlib import Path
import unittest

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.third_party.utils.protolib import decodeMessage

from astra.run import completed_ranks
from speculative.event_model import EventTrace, EventType
from speculative.speculative_decode import build_speculative_iteration
from tracegen.spec_to_chakra import convert_trace
from workloads.prefill_decode import build_prefill_decode


class WorkloadTests(unittest.TestCase):
    def test_autoregressive_steps_wait_for_feedback(self):
        trace = build_prefill_decode(decode_tokens=3)
        by_id = {event.event_id: event for event in trace.events}
        for event in trace.events:
            if event.rank == 0 and event.event_type is EventType.COMPUTE and event.phase == "decode":
                previous = by_id[event.dependencies[0]]
                self.assertIs(previous.event_type, EventType.RECV)
                self.assertEqual(previous.src, 1)
                self.assertEqual(previous.bytes, 4)
        sends = [event for event in trace.events if event.event_type is EventType.SEND and event.src == 0]
        self.assertEqual([event.bytes for event in sends], [1048576, 4096, 4096, 4096])

    def test_json_and_chakra_preserve_dependencies_phases_and_payloads(self):
        for trace in (build_prefill_decode(), build_speculative_iteration()):
            with self.subTest(parameters=trace.parameters), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                restored = EventTrace.read_json(trace.write_json(root / "events.json"))
                self.assertEqual(restored.to_dict(), trace.to_dict())
                outputs = convert_trace(restored, root / "workload")
                self.assertEqual(len(outputs), 2)
                for rank, path in enumerate(outputs):
                    with path.open("rb") as stream:
                        metadata = GlobalMetadata()
                        self.assertTrue(decodeMessage(stream, metadata))
                        nodes = []
                        while True:
                            node = Node()
                            if not decodeMessage(stream, node):
                                break
                            nodes.append(node)
                    expected = sorted((event for event in trace.events if event.rank == rank), key=lambda event: event.event_id)
                    self.assertEqual(len(nodes), len(expected))
                    for node, event in zip(nodes, expected):
                        self.assertEqual(node.id, event.event_id)
                        self.assertEqual(list(node.data_deps), event.dependencies)
                        attrs = {attr.name: attr for attr in node.attr}
                        if event.phase:
                            self.assertEqual(attrs["phase"].string_val, event.phase)
                        if event.event_type is EventType.COMPUTE:
                            self.assertEqual(node.duration_micros, event.duration_us)
                        else:
                            self.assertEqual(attrs["comm_size"].uint64_val, event.bytes)
                            self.assertEqual(attrs["comm_src"].uint32_val, event.src)
                            self.assertEqual(attrs["comm_dst"].uint32_val, event.dst)

    def test_invalid_workloads_fail_before_encoding(self):
        for kwargs in ({"decode_tokens": 0}, {"decode_bytes": 0}, {"prefill_compute_us": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_prefill_decode(**kwargs)
        trace = build_prefill_decode()
        trace.events[2].bytes += 1
        with self.assertRaisesRegex(ValueError, "unmatched communication"):
            trace.validate()

    def test_incomplete_simulation_is_not_success(self):
        first = "sys[0] finished, 901000 cycles, exposed communication 501000 cycles.\n"
        with self.assertRaises(RuntimeError):
            completed_ranks(first, 2)
        with self.assertRaises(RuntimeError):
            completed_ranks("", 2)
        ranks = completed_ranks(first + "sys[1] finished, 901000 cycles, exposed communication 401000 cycles.\n"
                                + "sys[0], GPU time: 400000\n", 2)
        self.assertEqual(ranks["0"]["compute_cycles"], 400000)
        self.assertEqual(ranks["1"]["cycles"], 901000)


if __name__ == "__main__":
    unittest.main()
