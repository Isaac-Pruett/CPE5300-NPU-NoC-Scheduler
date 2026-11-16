"""Single-resource latency improves without claiming a makespan improvement."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from astra.configuration import load_config
from astra.run import BINARIES, apply_schedule
from schedulers.fifo import FIFO
from schedulers.ljf import LongestJobFirst
from schedulers.sjf import ShortestJobFirst
from speculative.event_model import EventType
from workloads.single_processor import (
    build_single_processor_demo,
    single_processor_metrics,
)


class SingleProcessorTests(unittest.TestCase):
    def test_independent_jobs_and_exact_analytical_latency(self):
        trace = build_single_processor_demo()
        self.assertEqual(trace.num_ranks, 1)
        self.assertTrue(
            all(
                event.event_type is EventType.COMPUTE and not event.dependencies
                for event in trace.events
            )
        )
        fifo = apply_schedule(trace, FIFO)
        sjf = apply_schedule(trace, ShortestJobFirst)
        self.assertEqual(fifo.parameters["compute_schedule"], {"0": [0, 1, 2]})
        self.assertEqual(sjf.parameters["compute_schedule"], {"0": [2, 1, 0]})
        fifo_metrics = single_processor_metrics(fifo)
        sjf_metrics = single_processor_metrics(sjf)
        self.assertEqual(fifo_metrics["mean_completion_us"], 1070)
        self.assertEqual(sjf_metrics["mean_completion_us"], 410)
        self.assertEqual(fifo_metrics["mean_waiting_us"], 700)
        self.assertEqual(sjf_metrics["mean_waiting_us"], 40)
        self.assertEqual(fifo_metrics["makespan_us"], 1110)
        self.assertEqual(sjf_metrics["makespan_us"], 1110)
        self.assertEqual(
            [task["completion_us"] for task in sjf_metrics["tasks"]], [10, 110, 1110]
        )
        self.assertEqual(
            sjf_metrics["source"],
            "analytical_from_trace_durations_not_astra_timestamps",
        )

    def test_generated_jobs_are_seeded_independent_and_bounded(self):
        trace = build_single_processor_demo(jobs=1000, seed=42)
        self.assertEqual(len(trace.events), 1000)
        self.assertEqual(
            trace.to_dict(), build_single_processor_demo(jobs=1000, seed=42).to_dict()
        )
        self.assertNotEqual(
            trace.parameters["job_durations_us"],
            build_single_processor_demo(jobs=1000, seed=43).parameters[
                "job_durations_us"
            ],
        )
        self.assertEqual(trace.parameters["seed"], 42)
        self.assertTrue(
            all(
                not event.dependencies
                and event.rank == 0
                and 10 <= event.duration_us <= 1000
                for event in trace.events
            )
        )

    def test_generated_trace_latency_ordering_and_equal_work(self):
        trace = build_single_processor_demo(jobs=100, seed=42)
        metrics = [
            single_processor_metrics(apply_schedule(trace, policy))
            for policy in (ShortestJobFirst, FIFO, LongestJobFirst)
        ]
        self.assertLess(
            metrics[0]["mean_completion_us"], metrics[1]["mean_completion_us"]
        )
        self.assertLess(
            metrics[1]["mean_completion_us"], metrics[2]["mean_completion_us"]
        )
        self.assertEqual(
            {metric["makespan_us"] for metric in metrics},
            {sum(trace.parameters["job_durations_us"])},
        )

    def test_invalid_generated_job_counts_are_rejected(self):
        for jobs in (0, -1, True, 1.5):
            with (
                self.subTest(jobs=jobs),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                build_single_processor_demo(jobs=jobs)

    def test_metrics_require_explicit_complete_valid_order(self):
        trace = build_single_processor_demo()
        with self.assertRaisesRegex(ValueError, "complete explicit"):
            single_processor_metrics(trace)
        trace = apply_schedule(trace, FIFO)
        trace.parameters["compute_schedule"]["0"] = [0, 0, 2]
        with self.assertRaisesRegex(ValueError, "complete explicit"):
            single_processor_metrics(trace)
        trace.parameters["compute_schedule"]["0"] = [2, 1, 0]
        with self.assertRaisesRegex(ValueError, "violates"):
            single_processor_metrics(trace)

    def test_jetson_is_singleton_and_rejects_two_rank_workloads(self):
        self.assertEqual(load_config("jetson_orin_nano", num_ranks=1).num_ranks, 1)
        with self.assertRaisesRegex(ValueError, "requires 2"):
            load_config("jetson_orin_nano", num_ranks=2)

    @unittest.skipUnless(
        all(shutil.which(binary) for binary in BINARIES.values()),
        "requires both container ASTRA binaries",
    )
    def test_one_rank_completes_with_equal_makespan_under_both_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend in BINARIES:
                summaries = {}
                for policy in ("fifo", "sjf"):
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
                    self.assertEqual(set(summary["ranks"]), {"0"})
                    self.assertEqual(summary["hardware"]["num_ranks"], 1)
                    summaries[policy] = summary
                self.assertEqual(
                    summaries["fifo"]["ranks"]["0"]["cycles"],
                    summaries["sjf"]["ranks"]["0"]["cycles"],
                )
                self.assertEqual(
                    summaries["fifo"]["ranks"]["0"]["compute_cycles"], 1110000
                )
                self.assertLess(
                    summaries["sjf"]["single_processor_metrics"]["mean_completion_us"],
                    summaries["fifo"]["single_processor_metrics"]["mean_completion_us"],
                )


if __name__ == "__main__":
    unittest.main()
