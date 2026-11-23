"""Keep ASTRA measurements distinct from optional analytical latency reports."""

import copy
import io
import itertools
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

    def with_simulated_latency(self, run, analytical=False):
        self.with_latency(run)
        metrics = copy.deepcopy(run.summary["single_processor_metrics"])
        metrics.update(
            source="astra_sim_callbacks",
            mean_completion_us=14.5,
            mean_waiting_us=7.0,
            makespan_us=17.0,
        )
        metrics["tasks"][0].update(start_us=2.0, completion_us=12.0, waiting_us=2.0)
        metrics["tasks"][1].update(start_us=12.0, completion_us=17.0, waiting_us=12.0)
        run.summary["simulated_single_processor_metrics"] = metrics
        if not analytical:
            run.summary.pop("single_processor_metrics")
        return run

    def test_simulated_latency_and_jobs_survive_loading_separate_from_reference(self):
        run = self.with_simulated_latency(
            load_run(self.make_run("actual")), analytical=True
        )
        (run.directory / "summary.json").write_text(json.dumps(run.summary))
        (run.directory / "events.json").write_text(json.dumps(run.events))
        run = load_run(run.directory)
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], [])
        report = output.getvalue()
        simulated, analytical = report.split("ANALYTICAL single-processor latency", 1)
        self.assertIn("SIMULATED single-processor latency", simulated)
        self.assertIn("actual ASTRA callback measurements", simulated)
        self.assertIn("astra_sim_callbacks", simulated)
        self.assertIn("ns to us", simulated)
        self.assertIn("all jobs arriving at time 0", simulated)
        self.assertIn("not an analytical estimate", simulated)
        for value in ("14.500", "7.000", "17.000"):
            self.assertIn(value, simulated)
        self.assertNotIn("12.500", simulated)
        self.assertIn("12.500", analytical)
        self.assertIn("not ASTRA timestamps or measurements", analytical)
        tasks = simulated.split("SIMULATED per-job latency (us):", 1)[1]
        self.assertIn("Event ID", tasks)
        self.assertIn("Duration (us)", tasks)
        self.assertIn("Start (us)", tasks)
        self.assertIn("Completion (us)", tasks)
        self.assertIn("Waiting (us)", tasks)
        rows = [line.split("|") for line in tasks.splitlines() if run.label in line]
        self.assertEqual(
            [[cell.strip() for cell in row] for row in rows],
            [
                [run.label, "0", "first", "10.000", "2.000", "12.000", "2.000"],
                [run.label, "1", "second", "5.000", "12.000", "17.000", "12.000"],
            ],
        )

    def test_simulated_only_and_mixed_report_preserves_warnings(self):
        actual = self.with_simulated_latency(load_run(self.make_run("actual")))
        legacy = load_run(self.make_run("legacy", 50))
        warnings = comparison_warnings([actual, legacy])
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([actual, legacy], warnings)
        report = output.getvalue()
        section = report.split("SIMULATED single-processor latency", 1)[1].split(
            "Offline compute orders", 1
        )[0]
        self.assertIn(actual.label, section)
        self.assertNotIn(legacy.label, section)
        self.assertNotIn("ANALYTICAL", report)
        self.assertIn("WARNING: Runs differ in rank count.", report)
        self.assertNotIn("2.000x", report)

    def test_equal_completion_explanation_covers_each_latency_source(self):
        for source in ("simulated", "analytical", "both"):
            with self.subTest(source=source):
                runs = [
                    load_run(self.make_run(f"{source}-{index}")) for index in range(2)
                ]
                for run in runs:
                    if source == "analytical":
                        self.with_latency(run)
                    else:
                        self.with_simulated_latency(run, analytical=source == "both")
                output = io.StringIO()
                with redirect_stdout(output):
                    print_report(runs, [])
                explanation = output.getvalue().split(
                    "Total completion cycles are identical;", 1
                )[1]
                self.assertIn("without changing total runtime", explanation)
                if source != "analytical":
                    self.assertIn("actual ASTRA callbacks, not estimates", explanation)
                if source != "simulated":
                    self.assertIn("separate trace-duration reference", explanation)

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
        self.assertNotIn("SIMULATED", output.getvalue())
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
        for states in itertools.product(
            ("none", "analytical", "simulated", "both"), repeat=2
        ):
            with self.subTest(states=states):
                runs = [
                    load_run(self.make_run(f"chart-{states}-{index}", cycles))
                    for index, cycles in enumerate((100, 50))
                ]
                for run, state in zip(runs, states):
                    if state == "analytical":
                        self.with_latency(run)
                    elif state in ("simulated", "both"):
                        self.with_simulated_latency(run, analytical=state == "both")
                all_simulated = all(state in ("simulated", "both") for state in states)
                all_analytical_only = all(state == "analytical" for state in states)
                has_panel = all_simulated or all_analytical_only
                matplotlib = MagicMock()
                pyplot = matplotlib.pyplot
                figure, axis, latency_axis = MagicMock(), MagicMock(), MagicMock()
                pyplot.subplots.return_value = (
                    figure,
                    (axis, latency_axis) if has_panel else axis,
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
                if has_panel:
                    self.assertEqual(pyplot.subplots.call_args.args, (2, 1))
                    values = [14.5, 14.5] if all_simulated else [12.5, 12.5]
                    latency_axis.bar.assert_called_once_with([0, 1], values)
                    latency_axis.set_ylabel.assert_called_once_with(
                        "Simulated mean completion (us)"
                        if all_simulated
                        else "Analytical mean completion (us)"
                    )
                    title = latency_axis.set_title.call_args.args[0]
                    if all_simulated:
                        self.assertIn("SIMULATED actual ASTRA callbacks", title)
                        self.assertIn("astra_sim_callbacks", title)
                        self.assertIn("ns to us", title)
                        self.assertNotIn("ANALYTICAL", title)
                    else:
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

    def with_release_latency(self, run, key="simulated_single_processor_metrics"):
        if key == "simulated_single_processor_metrics":
            self.with_simulated_latency(run)
        else:
            self.with_latency(run)
        metrics = run.summary[key]
        metrics.update(
            mean_completion_us=122.0,
            mean_flow_us=11.0,
            mean_response_us=6.0,
            mean_waiting_us=6.0,
            makespan_us=125.0,
            arrival_assumption="declared workload releases, not observed callbacks",
        )
        metrics["tasks"][0].update(
            arrival_us=110.0,
            duration_us=6.0,
            start_us=119.0,
            completion_us=125.0,
            waiting_us=9.0,
            flow_us=15.0,
            response_us=9.0,
        )
        metrics["tasks"][1].update(
            arrival_us=112.0,
            duration_us=4.0,
            start_us=115.0,
            completion_us=119.0,
            waiting_us=3.0,
            flow_us=7.0,
            response_us=3.0,
        )
        return run

    def test_release_latency_report_prefers_flow_not_absolute_completion(self):
        run = self.with_release_latency(load_run(self.make_run("released")))
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], [])
        section = output.getvalue().split("SIMULATED single-processor latency", 1)[1]
        self.assertIn("Mean flow (us)", section)
        self.assertIn("11.000", section)
        self.assertNotIn("122.000", section)
        self.assertIn("125.000", section)  # Absolute makespan is retained.
        self.assertIn("not observed callbacks", section)
        self.assertIn("Arrival (us)", section)
        self.assertIn("Flow (us)", section)
        self.assertIn("Response (us)", section)
        self.assertIn("110.000", section)
        self.assertIn("15.000", section)

    def test_mixed_legacy_and_flow_tables_identify_different_bases(self):
        modern = self.with_release_latency(load_run(self.make_run("modern")))
        legacy = self.with_simulated_latency(load_run(self.make_run("legacy")))
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([modern, legacy], [])
        report = output.getvalue()
        self.assertIn("Mean flow / legacy completion (us)", report)
        self.assertIn("flow = completion - workload release", report)
        self.assertIn(
            "legacy absolute completion; assumes all jobs arriving at time 0", report
        )
        self.assertIn("11.000", report)
        self.assertIn("14.500", report)
        self.assertNotIn("122.000", report)

    def test_chart_uses_flow_for_both_sources_and_omits_mixed_semantics(self):
        for key in ("simulated_single_processor_metrics", "single_processor_metrics"):
            for mixed in (False, True):
                with self.subTest(key=key, mixed=mixed):
                    runs = [
                        self.with_release_latency(
                            load_run(self.make_run(f"flow-{key}-{mixed}-{index}")), key
                        )
                        for index in range(2)
                    ]
                    if mixed:
                        runs[1].summary[key].pop("mean_flow_us")
                    matplotlib = MagicMock()
                    pyplot = matplotlib.pyplot
                    figure, axis, latency_axis = MagicMock(), MagicMock(), MagicMock()
                    pyplot.subplots.return_value = (
                        figure,
                        axis if mixed else (axis, latency_axis),
                    )
                    with patch.dict(
                        "sys.modules",
                        {
                            "matplotlib": matplotlib,
                            "matplotlib.pyplot": pyplot,
                        },
                    ):
                        save_chart(runs, self.root / "flow.png", [])
                    if mixed:
                        self.assertEqual(pyplot.subplots.call_args.args, ())
                        latency_axis.bar.assert_not_called()
                    else:
                        latency_axis.bar.assert_called_once_with([0, 1], [11.0, 11.0])
                        self.assertIn(
                            "mean flow", latency_axis.set_ylabel.call_args.args[0]
                        )
                        title = latency_axis.set_title.call_args.args[0]
                        self.assertIn("Declared workload releases", title)
                        self.assertNotIn("all jobs arriving at time 0", title)

    def test_analytical_release_report_also_prefers_flow(self):
        run = self.with_release_latency(
            load_run(self.make_run("analytical-release")), "single_processor_metrics"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([run], [])
        section = output.getvalue().split("ANALYTICAL single-processor latency", 1)[1]
        self.assertIn("Mean flow (us)", section)
        self.assertIn("11.000", section)
        self.assertNotIn("122.000", section)

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
