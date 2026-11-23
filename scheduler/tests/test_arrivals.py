"""Release-aware queue behavior, encoding, and simulator-enforced idle gaps."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from astra.run import BINARIES, apply_schedule
from schedulers.fifo import FIFO
from schedulers.ljf import LongestJobFirst
from schedulers.sjf import ShortestJobFirst
from schedulers.task import Task
from speculative.event_model import Event, EventTrace, EventType
from workloads.single_processor import (
    build_single_processor_demo,
    single_processor_metrics,
)


class ArrivalPlanningTests(unittest.TestCase):
    def test_running_job_is_not_preempted_and_future_jobs_are_not_selected(self):
        trace = build_single_processor_demo(arrival_times_us=[0, 100, 200])
        fifo = apply_schedule(trace, FIFO)
        sjf = apply_schedule(trace, ShortestJobFirst)
        self.assertEqual(fifo.parameters["compute_schedule"], {"0": [0, 1, 2]})
        self.assertEqual(sjf.parameters["compute_schedule"], {"0": [0, 2, 1]})
        metrics = single_processor_metrics(sjf)
        self.assertEqual(
            [task["start_us"] for task in metrics["tasks"]], [0, 1000, 1010]
        )
        self.assertEqual(metrics["mean_flow_us"], 940)
        self.assertEqual(metrics["mean_waiting_us"], 570)
        self.assertEqual([event.dependencies for event in trace.events], [[], [], []])

    def test_unsorted_releases_preserve_fifo_arrival_order(self):
        trace = build_single_processor_demo(arrival_times_us=[500, 0, 50])
        for policy in (FIFO, ShortestJobFirst, LongestJobFirst):
            with self.subTest(policy=policy.__name__):
                scheduled = apply_schedule(trace, policy)
                self.assertEqual(
                    scheduled.parameters["compute_schedule"], {"0": [1, 2, 0]}
                )
                self.assertEqual(
                    [
                        task["start_us"]
                        for task in single_processor_metrics(scheduled)["tasks"]
                    ],
                    [0, 100, 500],
                )

    def test_arrivals_at_completion_boundary_are_eligible(self):
        trace = build_single_processor_demo(arrival_times_us=[0, 1000, 1000])
        self.assertEqual(
            apply_schedule(trace, ShortestJobFirst).parameters["compute_schedule"]["0"],
            [0, 2, 1],
        )

    def test_one_persistent_policy_dequeues_jobs_without_marking_them_finished(self):
        instances = []
        admitted = []

        class ObservedFIFO(FIFO):
            def __init__(self):
                super().__init__()
                instances.append(self)

            def admit(self, tasks):
                admitted.extend(tasks)
                super().admit(tasks)

        apply_schedule(
            build_single_processor_demo(arrival_times_us=[0, 100, 200]), ObservedFIFO
        )
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(admitted), 3)
        self.assertTrue(all(not task.finished for task in admitted))
        self.assertEqual(instances[0].queue, [])

    def test_direct_repeated_selection_returns_each_task_once(self):
        expected = {
            FIFO: [0, 1, 2],
            ShortestJobFirst: [2, 1, 0],
            LongestJobFirst: [0, 1, 2],
        }
        for policy_type, order in expected.items():
            with self.subTest(policy=policy_type.__name__):
                policy = policy_type()
                tasks = [
                    Task(event=event) for event in build_single_processor_demo().events
                ]
                policy.admit(tasks)
                self.assertEqual([policy.next().event.event_id for _ in tasks], order)
                with self.assertRaises(IndexError):
                    policy.next()
                self.assertTrue(all(not task.finished for task in tasks))

    def test_requeueing_policy_is_rejected(self):
        class BrokenFIFO(FIFO):
            def next(self):
                return self.queue[0]

        with self.assertRaisesRegex(ValueError, "dequeue"):
            apply_schedule(build_single_processor_demo(), BrokenFIFO)

    def test_nonzero_multi_rank_arrivals_are_not_silently_misplanned(self):
        trace = EventTrace(
            2,
            [
                Event(0, EventType.COMPUTE, 0, "a", duration_us=1, arrival_us=100),
                Event(1, EventType.COMPUTE, 1, "b", duration_us=1),
            ],
        )
        with self.assertRaisesRegex(ValueError, "single-rank"):
            apply_schedule(trace, FIFO)

    def test_arrival_roundtrip_validation_and_chakra_attribute(self):
        from tracegen.spec_to_chakra import _convert_event

        event = Event(0, EventType.COMPUTE, 0, "a", duration_us=1, arrival_us=123)
        event.validate(1)
        self.assertEqual(Event.from_dict(event.to_dict()).arrival_us, 123)
        attrs = {attr.name: attr for attr in _convert_event(event, {}, 0).attr}
        self.assertEqual(attrs["arrival_us"].uint64_val, 123)
        self.assertEqual(
            Event.from_dict(
                {
                    "event_id": 1,
                    "event_type": "COMPUTE",
                    "rank": 0,
                    "name": "legacy",
                    "duration_us": 1,
                }
            ).arrival_us,
            0,
        )
        for invalid in (-1, True, 1.5, (2**64 - 1) // 1000 + 1):
            event.arrival_us = invalid
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "arrival_us"),
            ):
                event.validate(1)


class ArrivalSimulationTests(unittest.TestCase):
    @unittest.skipUnless(
        all(shutil.which(binary) for binary in BINARIES.values()),
        "requires both container ASTRA binaries",
    )
    def test_release_wakeups_enforce_initial_and_mid_run_idle_on_both_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend in BINARIES:
                for policy in ("fifo", "sjf", "ljf"):
                    with self.subTest(backend=backend, policy=policy):
                        output = Path(directory) / backend / policy
                        result = subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "astra.run",
                                "--workload",
                                "single-processor-demo",
                                "--config-dir",
                                "jetson_orin_nano",
                                "--backend",
                                backend,
                                "--policy",
                                policy,
                                "--arrival-times-us",
                                "500",
                                "2000",
                                "2000",
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
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        summary = json.loads((output / "summary.json").read_text())
                        self.assertEqual(summary["status"], "completed")
                        self.assertIn("arrivals-v1", summary["astra_sim_revision"])
                        with (output / "event_timing.csv").open() as stream:
                            rows = {
                                int(row["event_id"]): row
                                for row in csv.DictReader(stream)
                            }
                        self.assertEqual(float(rows[0]["start_us"]), 500)
                        self.assertEqual(float(rows[0]["end_us"]), 1500)
                        self.assertTrue(
                            all(float(rows[i]["start_us"]) >= 2000 for i in (1, 2))
                        )
                        # Idle time is not implemented as fake GPU work.
                        self.assertEqual(
                            summary["ranks"]["0"]["compute_cycles"], 1110000
                        )
                        self.assertEqual(summary["ranks"]["0"]["cycles"], 2110000)
                        metrics = summary["simulated_single_processor_metrics"]
                        self.assertEqual(metrics["makespan_us"], 2110)
                        self.assertLess(
                            metrics["mean_flow_us"], metrics["mean_completion_us"]
                        )
                        for task in metrics["tasks"]:
                            self.assertGreaterEqual(
                                task["start_us"], task["arrival_us"]
                            )
                            self.assertEqual(
                                task["flow_us"],
                                task["completion_us"] - task["arrival_us"],
                            )


if __name__ == "__main__":
    unittest.main()
