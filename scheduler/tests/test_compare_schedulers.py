"""All-policy batches preserve inputs, discover policies, and fail safely."""

import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from astra.run import BINARIES
from experiments.compare_schedulers import main, run_experiment


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "batch with spaces"
        self.commands = []

    def simulate(self, command, **kwargs):
        self.commands.append(command)
        policy = command[command.index("--policy") + 1]
        output = Path(command[command.index("--output-dir") + 1])
        snapshot = Path(command[command.index("--config-dir") + 1])
        output.mkdir()
        for path in snapshot.iterdir():
            shutil.copyfile(path, output / path.name)
        summary = {
            "status": "completed",
            "workload": "single-processor-demo",
            "backend": "congestion-aware",
            "timing_source": "synthetic",
            "astra_sim_revision": "test-revision",
            "scheduling_policy": policy,
            "parameters": {"compute_schedule": {"0": [0]}, "scheduling_policy": policy},
            "ranks": {"0": {"cycles": 100}},
        }
        (output / "summary.json").write_text(json.dumps(summary))
        (output / "events.json").write_text(
            json.dumps(
                {
                    "num_ranks": 1,
                    "events": [{"event_id": 0, "name": "job"}],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    def test_all_discovered_policies_share_inputs_and_fifo_is_baseline(self):
        discovered = {
            "sjf": object,
            "new-policy": object,
            "fifo": object,
            "ljf": object,
        }
        with (
            patch(
                "experiments.compare_schedulers.discover_schedulers",
                return_value=discovered,
            ),
            patch(
                "experiments.compare_schedulers.subprocess.run",
                side_effect=self.simulate,
            ),
            patch("experiments.compare_schedulers.save_chart") as chart,
            patch("experiments.compare_schedulers.save_gantt") as gantt,
            redirect_stdout(io.StringIO()),
        ):
            result = run_experiment(
                output_dir=self.output, runner_args=["--decode-tokens", "8"]
            )
        self.assertEqual(result, self.output)
        manifest = json.loads((result / "batch.json").read_text())
        expected = ["fifo", "ljf", "new-policy", "sjf"]
        self.assertEqual(manifest["policies"], expected)
        self.assertEqual(manifest["completed_policies"], expected)
        self.assertEqual(manifest["status"], "completed")
        snapshots = set()
        for command in self.commands:
            self.assertEqual(
                command[command.index("--workload") + 1], "single-processor-demo"
            )
            snapshots.add(command[command.index("--config-dir") + 1])
            self.assertEqual(command[-2:], ["--decode-tokens", "8"])
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(Path(snapshots.pop()).name, "jetson_orin_nano")
        chart.assert_called_once()
        self.assertEqual(
            [run.summary["scheduling_policy"] for run in chart.call_args.args[0]],
            expected,
        )
        self.assertEqual(chart.call_args.args[1], result / "comparison.png")
        gantt.assert_called_once_with(
            [result / name for name in expected], result / "gantt.png"
        )

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing data")
        with patch("experiments.compare_schedulers.subprocess.run") as child:
            with self.assertRaises(FileExistsError):
                run_experiment(output_dir=self.output)
        child.assert_not_called()
        self.assertEqual(sentinel.read_text(), "existing data")

    def test_failed_policy_stops_batch_without_partial_chart(self):
        def fail_second(command, **kwargs):
            if self.commands:
                self.commands.append(command)
                return subprocess.CompletedProcess(command, 1)
            return self.simulate(command, **kwargs)

        with (
            patch(
                "experiments.compare_schedulers.discover_schedulers",
                return_value={"fifo": object, "bad": object, "later": object},
            ),
            patch(
                "experiments.compare_schedulers.subprocess.run", side_effect=fail_second
            ),
            patch("experiments.compare_schedulers.save_chart") as chart,
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "bad.*failed"),
        ):
            run_experiment(output_dir=self.output)
        manifest = json.loads((self.output / "batch.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["completed_policies"], ["fifo"])
        self.assertEqual(len(self.commands), 2)
        chart.assert_not_called()

    def test_incomplete_zero_exit_run_is_not_compared(self):
        def incomplete(command, **kwargs):
            result = self.simulate(command, **kwargs)
            output = Path(command[command.index("--output-dir") + 1])
            path = output / "summary.json"
            summary = json.loads(path.read_text())
            summary["status"] = "running"
            path.write_text(json.dumps(summary))
            return result

        with (
            patch(
                "experiments.compare_schedulers.subprocess.run", side_effect=incomplete
            ),
            patch("experiments.compare_schedulers.save_chart") as chart,
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "not completed"),
        ):
            run_experiment(output_dir=self.output)
        chart.assert_not_called()
        self.assertEqual(len(self.commands), 1)

    def test_forwarding_cannot_override_managed_options_or_abbreviations(self):
        for option in (
            "--policy=sjf",
            "--pol",
            "--output-dir",
            "--config-dir=x",
            "--list-policies",
            "--workload",
            "--backend",
            "--timeout",
        ):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, "managed"),
            ):
                run_experiment(output_dir=self.output, runner_args=[option])
        self.assertFalse(self.output.exists())

    def test_empty_discovery_and_invalid_timeout_do_not_create_outputs(self):
        with patch(
            "experiments.compare_schedulers.discover_schedulers", return_value={}
        ):
            with self.assertRaisesRegex(ValueError, "no schedulers"):
                run_experiment(output_dir=self.output)
        for timeout in (0, -1, float("nan"), float("inf")):
            with (
                self.subTest(timeout=timeout),
                self.assertRaisesRegex(ValueError, "timeout"),
            ):
                run_experiment(output_dir=self.output, timeout=timeout)
        self.assertFalse(self.output.exists())

    def test_cli_overrides_and_forwarding(self):
        with (
            patch(
                "sys.argv",
                [
                    "compare_schedulers",
                    "--experiment",
                    "h100",
                    "--workload",
                    "prefill-decode",
                    "--config-dir",
                    "h100_sxm_pair",
                    "--backend",
                    "congestion-unaware",
                    "--",
                    "--decode-tokens",
                    "8",
                ],
            ),
            patch("experiments.compare_schedulers.run_experiment") as experiment,
        ):
            main()
        self.assertEqual(experiment.call_args.args, ("h100",))
        self.assertEqual(experiment.call_args.kwargs["workload"], "prefill-decode")
        self.assertEqual(experiment.call_args.kwargs["backend"], "congestion-unaware")
        self.assertEqual(
            experiment.call_args.kwargs["runner_args"], ["--decode-tokens", "8"]
        )


class BatchSimulationTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which(BINARIES["congestion-aware"]), "requires container ASTRA binary"
    )
    def test_both_named_experiments_complete_and_generate_png(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("jetson", "h100"):
                with self.subTest(experiment=name), redirect_stdout(io.StringIO()):
                    output = run_experiment(
                        name, output_dir=Path(directory) / name, timeout=30
                    )
                    manifest = json.loads((output / "batch.json").read_text())
                    self.assertEqual(manifest["status"], "completed")
                    self.assertEqual(
                        manifest["completed_policies"], manifest["policies"]
                    )
                    self.assertEqual(manifest["policies"][0], "fifo")
                    self.assertEqual(manifest["comparison_warnings"], [])
                    self.assertTrue(
                        (output / "comparison.png")
                        .read_bytes()
                        .startswith(b"\x89PNG\r\n\x1a\n")
                    )
                    self.assertTrue(
                        (output / "gantt.png")
                        .read_bytes()
                        .startswith(b"\x89PNG\r\n\x1a\n")
                    )
                    for policy in manifest["policies"]:
                        summary = json.loads(
                            (output / policy / "summary.json").read_text()
                        )
                        self.assertEqual(summary["status"], "completed")
                        self.assertEqual(summary["scheduling_policy"], policy)


if __name__ == "__main__":
    unittest.main()
