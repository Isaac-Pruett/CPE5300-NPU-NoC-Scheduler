"""Comparison reports use recorded ASTRA data, not speculation estimates."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from compare_results import comparison_warnings, load_run, print_report


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
