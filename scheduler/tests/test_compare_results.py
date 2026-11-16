"""Keep ASTRA measurements distinct from optional analytical latency reports."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from compare_results import comparison_warnings, load_run, print_report, save_chart


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, name, cycles=100, policy="fifo"):
        directory = self.root / name
        directory.mkdir()
        summary = {
            "status": "completed",
            "workload": "example",
            "backend": "congestion-aware",
            "timing_source": "synthetic",
            "astra_sim_revision": "revision",
            "scheduling_policy": policy,
            "parameters": {
                "size": 10,
                "scheduling_policy": policy,
                "compute_schedule": {"0": [0], "1": [1]},
            },
            "ranks": {"0": {"cycles": cycles}, "1": {"cycles": cycles // 2}},
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "events.json").write_text(
            json.dumps(
                {
                    "num_ranks": 2,
                    "events": [
                        {"event_id": 0, "name": "first"},
                        {"event_id": 1, "name": "second"},
                    ],
                }
            )
        )
        for name in ("system.json", "network.yml", "remote_memory.json"):
            (directory / name).write_text("same configuration")
        return directory

    def test_completion_is_max_not_sum_and_policy_metadata_is_ignored(self):
        baseline = load_run(self.make_run("fifo", 100))
        candidate = load_run(self.make_run("sjf", 50, "sjf") / "summary.json")
        self.assertEqual(baseline.completion_cycles, 100)
        self.assertEqual(comparison_warnings([baseline, candidate]), [])
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([baseline, candidate], [])
        self.assertIn("2.000x", output.getvalue())
        self.assertIn("0:first", output.getvalue())
        self.assertIn("N/A", output.getvalue())  # Missing optional compute metrics.

    def test_long_compute_orders_are_previewed_without_modifying_saved_order(self):
        run = load_run(self.make_run("long"))
        run.summary["parameters"]["compute_schedule"] = {"0": list(range(100))}
        run.events["events"] = [
            {"event_id": index, "name": f"job_{index}"} for index in range(100)
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], [])
        self.assertIn("0:job_0", output.getvalue())
        self.assertIn("75 more", output.getvalue())
        self.assertNotIn("99:job_99", output.getvalue())
        self.assertEqual(len(run.summary["parameters"]["compute_schedule"]["0"]), 100)

    def with_latency(self, run):
        run.events["num_ranks"] = 1
        run.summary["ranks"].pop("1")
        run.summary["parameters"]["compute_schedule"] = {"0": [0, 1]}
        run.summary["single_processor_metrics"] = {
            "source": "analytical_from_trace_durations_not_astra_timestamps",
            "mean_completion_us": 12.5,
            "mean_waiting_us": 5.0,
            "makespan_us": 15.0,
            "tasks": [
                {
                    "event_id": 0,
                    "name": "first",
                    "duration_us": 10.0,
                    "start_us": 0.0,
                    "completion_us": 10.0,
                    "waiting_us": 0.0,
                },
                {
                    "event_id": 1,
                    "name": "second",
                    "duration_us": 5.0,
                    "start_us": 10.0,
                    "completion_us": 15.0,
                    "waiting_us": 10.0,
                },
            ],
        }
        return run

    def test_analytical_latency_is_separate_and_survives_loading(self):
        run = self.with_latency(load_run(self.make_run("fifo")))
        (run.directory / "summary.json").write_text(json.dumps(run.summary))
        (run.directory / "events.json").write_text(json.dumps(run.events))
        run = load_run(run.directory)
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], [])
        report = output.getvalue()
        self.assertIn("Completion cycles", report)
        self.assertIn("ANALYTICAL single-processor latency", report)
        self.assertIn("not ASTRA timestamps or measurements", report)
        self.assertIn("Independent COMPUTE jobs, all arriving at time 0", report)
        self.assertIn("one rank, serial schedule, zero overhead", report)
        for text in (
            "Mean completion (us)",
            "Mean waiting (us)",
            "Makespan (us)",
            "12.500",
            "5.000",
            "15.000",
        ):
            self.assertIn(text, report)
        self.assertEqual(run.completion_cycles, 100)

    def test_absent_latency_is_not_invented(self):
        run = load_run(self.make_run("legacy"))
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], [])
        self.assertNotIn("ANALYTICAL", output.getvalue())
        self.assertNotIn("Mean completion (us)", output.getvalue())
        self.assertNotIn("single_processor_metrics", run.summary)

    def test_mixed_latency_reports_only_available_metrics_and_keeps_warnings(self):
        available = self.with_latency(load_run(self.make_run("available")))
        missing = load_run(self.make_run("legacy", 50))
        warnings = comparison_warnings([available, missing])
        self.assertIn("Runs differ in rank count.", warnings)
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([available, missing], warnings)
        report = output.getvalue()
        analytical = report.split("ANALYTICAL single-processor latency", 1)[1].split(
            "Offline compute orders", 1
        )[0]
        self.assertIn(available.label, analytical)
        self.assertNotIn(missing.label, analytical)
        self.assertIn("WARNING: Runs differ in rank count.", report)
        self.assertNotIn("2.000x", report)

    def test_chart_adds_latency_subplot_only_when_all_runs_have_metrics(self):
        for present in ((False, False), (True, False), (True, True)):
            with self.subTest(present=present):
                runs = [
                    load_run(self.make_run(f"chart-{present}-{index}", cycles))
                    for index, cycles in enumerate((100, 50))
                ]
                for run, has_metrics in zip(runs, present):
                    if has_metrics:
                        self.with_latency(run)
                matplotlib = MagicMock()
                pyplot = matplotlib.pyplot
                figure, axis, latency_axis = MagicMock(), MagicMock(), MagicMock()
                pyplot.subplots.return_value = (
                    figure,
                    (axis, latency_axis) if all(present) else axis,
                )
                with patch.dict(
                    "sys.modules",
                    {
                        "matplotlib": matplotlib,
                        "matplotlib.pyplot": pyplot,
                    },
                ):
                    save_chart(
                        runs, self.root / "plots" / "comparison.png", ["confounder"]
                    )
                axis.bar.assert_called_once_with([0, 1], [100, 50])
                axis.set_title.assert_called_once_with("ASTRA-sim policy comparison")
                if all(present):
                    self.assertEqual(pyplot.subplots.call_args.args, (2, 1))
                    latency_axis.bar.assert_called_once_with([0, 1], [12.5, 12.5])
                    latency_axis.set_ylabel.assert_called_once_with(
                        "Analytical mean completion (us)"
                    )
                    title = latency_axis.set_title.call_args.args[0]
                    self.assertIn("ANALYTICAL", title)
                    self.assertIn("not ASTRA measurements", title)
                else:
                    self.assertEqual(pyplot.subplots.call_args.args, ())
                    latency_axis.bar.assert_not_called()
                self.assertIn("not verified comparable", figure.text.call_args.args[2])
                figure.savefig.assert_called_once_with(
                    self.root / "plots" / "comparison.png", dpi=180
                )
                pyplot.close.assert_called_once_with(figure)

    def test_incomplete_and_missing_rank_runs_are_rejected(self):
        for change in (
            lambda summary: summary.update(status="failed"),
            lambda summary: summary["ranks"].pop("1"),
            lambda summary: summary["ranks"]["0"].update(cycles=-1),
        ):
            directory = self.make_run(str(len(list(self.root.iterdir()))))
            path = directory / "summary.json"
            summary = json.loads(path.read_text())
            change(summary)
            path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_run(directory)

    def test_mismatched_parameters_and_configurations_suppress_speedup(self):
        baseline = load_run(self.make_run("fifo"))
        candidate = load_run(self.make_run("sjf", 50, "sjf"))
        candidate.summary["parameters"]["size"] = 20
        (candidate.directory / "network.yml").write_text("different")
        warnings = comparison_warnings([baseline, candidate])
        self.assertTrue(any("parameters" in warning for warning in warnings))
        self.assertTrue(any("network.yml" in warning for warning in warnings))
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([baseline, candidate], warnings)
        self.assertNotIn("2.000x", output.getvalue())

    def test_missing_config_is_flagged_and_no_schedule_is_not_invented(self):
        run = load_run(self.make_run("none", policy="none"))
        (run.directory / "system.json").unlink()
        run.summary["parameters"].pop("compute_schedule")
        warnings = comparison_warnings([run])
        self.assertTrue(any("Missing system.json" in warning for warning in warnings))
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], warnings)
        self.assertIn("No recorded compute schedule", output.getvalue())


if __name__ == "__main__":
    unittest.main()
