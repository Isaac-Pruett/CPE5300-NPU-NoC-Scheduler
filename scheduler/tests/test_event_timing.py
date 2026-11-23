"""Lifecycle timestamps must come from actual simulator callbacks, with full coverage."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from astra.event_timing import (
    parse_event_timings,
    simulated_single_processor_metrics,
    write_event_timings,
)
from speculative.event_model import Event, EventTrace, EventType


def record(event_id, stage, cycle, rank=0, period=1):
    return f"ASTRA_EVENT_TIMING_V1,{rank},{event_id},{stage},{cycle},{period}\n"


def interval(event_id, ready, start, end, rank=0, period=1):
    return "".join(
        record(event_id, stage, tick, rank, period)
        for stage, tick in (("ready", ready), ("start", start), ("end", end))
    )


class EventTimingTests(unittest.TestCase):
    def setUp(self):
        self.trace = EventTrace(
            1,
            [
                Event(0, EventType.COMPUTE, 0, "first", duration_us=1000),
                Event(1, EventType.COMPUTE, 0, "second", duration_us=1000),
            ],
        )
        self.ranks = {"0": {"cycles": 100}}
        self.output = (
            "other simulator logging\n"
            + interval(0, 0, 10, 50)
            + interval(1, 0, 50, 100)
        )

    def test_actual_timestamps_are_used_not_declared_durations(self):
        rows = parse_event_timings(self.output, self.trace, self.ranks)
        self.assertEqual(rows[0]["execution_us"], 0.04)
        self.assertEqual(rows[0]["queue_wait_us"], 0.01)
        self.assertEqual(rows[1]["end_us"], 0.1)
        self.trace.parameters["workload"] = "single_processor_demo"
        metrics = simulated_single_processor_metrics(rows, self.trace)
        self.assertAlmostEqual(metrics["mean_completion_us"], 0.075)
        self.assertAlmostEqual(metrics["mean_waiting_us"], 0.03)
        self.assertEqual(metrics["source"], "astra_sim_callbacks")
        self.assertEqual(metrics["mean_flow_us"], metrics["mean_completion_us"])
        self.assertEqual(metrics["mean_response_us"], metrics["mean_waiting_us"])
        self.assertTrue(all(row["arrival_cycle"] == 0 for row in rows))

    def test_late_releases_use_trace_ids_and_callback_durations(self):
        self.trace.parameters["workload"] = "single_processor_demo"
        self.trace.events[0].arrival_us = 10
        self.trace.events[1].arrival_us = 12
        # Deliberately dispatch event 1 first, not in trace order.
        output = interval(1, 12000, 15000, 19000) + interval(0, 10000, 19000, 25000)
        rows = parse_event_timings(output, self.trace, {"0": {"cycles": 25000}})
        metrics = simulated_single_processor_metrics(rows, self.trace)
        self.assertEqual(metrics["mean_completion_us"], 22)
        self.assertEqual(metrics["makespan_us"], 25)
        self.assertEqual(metrics["mean_flow_us"], 11)
        self.assertEqual(metrics["mean_response_us"], 6)
        self.assertEqual(metrics["mean_waiting_us"], 6)
        self.assertIn("workload releases", metrics["arrival_assumption"])
        tasks = {task["event_id"]: task for task in metrics["tasks"]}
        self.assertEqual(tasks[0]["arrival_us"], 10)
        self.assertEqual(tasks[0]["flow_us"], 15)
        self.assertEqual(tasks[0]["response_us"], 9)
        self.assertEqual(tasks[0]["duration_us"], 6)
        self.assertEqual(tasks[1]["arrival_us"], 12)
        self.assertEqual(tasks[1]["flow_us"], 7)
        self.assertEqual(tasks[1]["waiting_us"], 3)
        self.assertEqual(rows[0]["arrival_cycle"], 12000)

    def test_early_ready_or_dispatch_is_rejected(self):
        self.trace.events[0].arrival_us = 1
        for ready, start in ((999, 1000), (999, 999), (1000, 999)):
            with self.subTest(ready=ready, start=start):
                output = interval(0, ready, start, 1500) + interval(1, 0, 1500, 2000)
                with self.assertRaisesRegex(ValueError, "before its arrival"):
                    parse_event_timings(output, self.trace, {"0": {"cycles": 2000}})

    def test_arrival_clock_conversion_is_exact_integer_arithmetic(self):
        # Larger than float's exact integer range; no float may round the release.
        arrival = 2**53 + 1
        self.trace.events[0].arrival_us = arrival
        tick = arrival * 1000
        output = interval(0, tick, tick, tick) + interval(1, 0, 0, 0)
        rows = parse_event_timings(output, self.trace, {"0": {"cycles": tick}})
        self.assertEqual(rows[1]["arrival_cycle"], tick)
        early = interval(0, tick - 1, tick, tick) + interval(1, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "before its arrival"):
            parse_event_timings(early, self.trace, {"0": {"cycles": tick}})
        self.trace.events[0].arrival_us = 1
        output = interval(0, 334, 334, 334, period=3) + interval(1, 0, 0, 0, period=3)
        with self.assertRaisesRegex(ValueError, "exact clock cycle"):
            parse_event_timings(output, self.trace, {"0": {"cycles": 334}})

    def test_declared_arrivals_are_not_observed_ready_callbacks(self):
        self.trace.events[0].arrival_us = 2
        self.trace.events[1].arrival_us = 3
        self.trace.events[1].dependencies = [0]
        output = interval(0, 4000, 5000, 8000) + interval(1, 8000, 9000, 10000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, _ = write_event_timings(
                output, self.trace, {"0": {"cycles": 10000}}, root
            )
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["source"], "astra_sim_callbacks")
            self.assertEqual(metadata["arrival_source"], "workload_release_times")
            self.assertIn("released and dependency-free", metadata["ready_semantics"])
            timeline = json.loads((root / "timesteps.json").read_text())
            self.assertEqual(timeline["schema_version"], 2)
            self.assertEqual(timeline["arrival_source"], "workload_release_times")
            steps = {step["cycle"]: step for step in timeline["steps"]}
            self.assertEqual(steps[2000]["arrived"], [{"rank": 0, "event_id": 0}])
            self.assertEqual(steps[3000]["arrived"], [{"rank": 0, "event_id": 1}])
            self.assertEqual(steps[2000]["ready"], [])
            self.assertEqual(steps[4000]["arrived"], [])
            self.assertEqual(steps[4000]["ready"], [{"rank": 0, "event_id": 0}])
            with (root / "event_timing.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["arrival_us"], "2")
            self.assertEqual(saved[0]["arrival_cycle"], "2000")
            self.assertEqual(saved[0]["ready_cycle"], "4000")

    def test_clock_period_is_exported_not_guessed(self):
        output = interval(0, 0, 10, 50, period=2) + interval(1, 0, 50, 100, period=2)
        rows = parse_event_timings(output, self.trace, self.ranks)
        self.assertEqual(rows[0]["clock_period_ns"], 2)
        self.assertEqual(rows[0]["start_us"], 0.02)

    def test_missing_duplicate_unknown_and_malformed_records_fail(self):
        cases = [
            ("no timestamps", "rebuild"),
            (interval(0, 0, 10, 50), "incomplete"),
            (self.output + record(0, "end", 50), "duplicate"),
            (self.output + record(9, "ready", 0), "unexpected"),
            (self.output + "ASTRA_EVENT_TIMING_V1,broken\n", "malformed"),
            (self.output + record(0, "arrived", 0), "malformed"),
            (self.output + "ASTRA_EVENT_TIMING_V1,0,0,start,-1,1\n", "malformed"),
            (interval(0, 0, 10, 50, period=0), "clock"),
            (interval(0, 0, 10, 50) + interval(1, 0, 50, 100, period=2), "clock"),
            (interval(0, 20, 10, 50) + interval(1, 0, 50, 100), "non-monotonic"),
            (interval(0, 0, 10, 50) + interval(1, 0, 50, 101), "rank completion"),
            (interval(0, 0, 10, 60) + interval(1, 0, 50, 100), "overlapping"),
        ]
        for output, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                parse_event_timings(output, self.trace, self.ranks)

    def test_dependencies_must_complete_before_observed_readiness(self):
        self.trace.events[1].dependencies = [0]
        with self.assertRaisesRegex(ValueError, "dependency"):
            parse_event_timings(self.output, self.trace, self.ranks)
        output = interval(0, 0, 10, 50) + interval(1, 50, 50, 100)
        self.assertEqual(len(parse_event_timings(output, self.trace, self.ranks)), 2)

    def test_csv_and_timesteps_preserve_simultaneous_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, rows = write_event_timings(
                self.output, self.trace, self.ranks, root
            )
            self.assertEqual(metadata["event_count"], 2)
            with (root / "event_timing.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["start_cycle"], "10")
            self.assertEqual(saved[1]["end_cycle"], "100")
            steps = json.loads((root / "timesteps.json").read_text())["steps"]
            step = next(step for step in steps if step["cycle"] == 50)
            self.assertEqual(step["completed"], [{"rank": 0, "event_id": 0}])
            self.assertEqual(step["started"], [{"rank": 0, "event_id": 1}])
            self.assertEqual(len(rows), 2)

    @unittest.skipUnless(
        shutil.which("AstraSim_Analytical_Congestion_Aware"),
        "requires container ASTRA binary",
    )
    def test_simulator_observes_resource_queue_wait_without_policy_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "astra.run",
                    "--workload",
                    "policy-demo",
                    "--policy",
                    "none",
                    "--output-dir",
                    str(output),
                    "--timeout",
                    "30",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (output / "event_timing.csv").open() as stream:
                rows = {int(row["event_id"]): row for row in csv.DictReader(stream)}
            # Both rank-0 compute nodes are dependency-free at zero; node 1 waits
            # for the occupied resource, so observed ready and start differ.
            self.assertEqual(int(rows[1]["ready_cycle"]), 0)
            self.assertEqual(int(rows[1]["start_cycle"]), 1000000)
            self.assertEqual(float(rows[1]["queue_wait_us"]), 1000)
            self.assertEqual(len(rows), 8)

    def test_zero_length_callback_and_communication_intervals_are_not_fabricated(self):
        trace = EventTrace(
            2,
            [
                Event(0, EventType.SEND, 0, "send", src=0, dst=1, bytes=4, tag="x"),
                Event(1, EventType.RECV, 1, "recv", src=0, dst=1, bytes=4, tag="x"),
            ],
        )
        output = interval(0, 0, 10, 10) + interval(1, 0, 0, 90, rank=1)
        rows = parse_event_timings(
            output, trace, {"0": {"cycles": 10}, "1": {"cycles": 90}}
        )
        by_id = {row["event_id"]: row for row in rows}
        self.assertEqual(by_id[0]["execution_us"], 0)
        self.assertEqual(by_id[1]["end_cycle"], 90)
        with self.assertRaisesRegex(ValueError, "single-processor"):
            simulated_single_processor_metrics(rows, trace)


if __name__ == "__main__":
    unittest.main()
