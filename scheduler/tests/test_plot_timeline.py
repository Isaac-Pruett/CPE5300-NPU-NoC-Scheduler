"""Exercise CSV validation and plot the recorded callbacks, not a fake schedule."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plot_timeline import FIELDS, save_gantt

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "plots" / "timeline.png"

    def row(self, start=10, end=20, **updates):
        row = dict(
            zip(
                FIELDS,
                (
                    0,
                    1,
                    "actual callback",
                    "COMPUTE",
                    "prefill",
                    0,
                    start * 1000,
                    end * 1000,
                    1,
                    0,
                    start,
                    end,
                    start,
                    end - start,
                ),
            )
        )
        row.update(updates)
        return row

    def run_dir(self, rows, name="run", fields=FIELDS):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        with (directory / "event_timing.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return directory

    def capture(self, runs, **kwargs):
        if plt is None:
            self.skipTest("matplotlib is not installed")
        fig, ax = plt.subplots()
        with patch.object(plt, "subplots", return_value=(fig, ax)):
            save_gantt(runs, self.output, **kwargs)
        return ax

    def test_actual_intervals_shared_origin_and_overlapping_receive_sublanes(self):
        run = self.run_dir(
            [
                self.row(37, 43, event_type="SEND", phase=""),
                self.row(39, 60, event_type="RECV", phase="decode", rank=1),
                self.row(40, 50, event_type="RECV", phase="decode", rank=1),
                self.row(80, 90),
            ]
        )
        ax = self.capture([run])
        bars = ax.patches
        self.assertEqual(
            [(b.get_x(), b.get_width()) for b in bars],
            [(80, 10), (37, 6), (39, 21), (40, 10)],
        )
        self.assertNotEqual(bars[2].get_y(), bars[3].get_y())
        self.assertEqual(ax.get_xlim(), (0, 90))
        self.assertTrue(self.output.read_bytes().startswith(b"\x89PNG"))
        self.assertIn("displayed 4 / 4", ax.get_title())
        self.assertEqual(len(ax.texts), 4)
        self.assertEqual(len(ax.get_legend().get_texts()), 6)

    def test_multiple_runs_and_optional_summary(self):
        first = self.run_dir([self.row(30, 40)], "first")
        second = self.run_dir([self.row(70, 80)], "second")
        (second / "summary.json").write_text(json.dumps({"scheduling_policy": "fifo"}))
        ax = self.capture([first, second])
        labels = [label.get_text() for label in ax.get_yticklabels()]
        self.assertIn("first / rank 0 / COMPUTE", labels)
        self.assertIn("second (fifo) / rank 0 / COMPUTE", labels)
        self.assertEqual([bar.get_x() for bar in ax.patches], [30, 70])
        (second / "summary.json").write_text("malformed optional metadata")
        self.capture([second])

    def test_filter_before_cap_keeps_earliest_and_preserves_intervals(self):
        run = self.run_dir(
            [self.row(i, i + 2, event_id=i) for i in reversed(range(1200))]
        )
        ax = self.capture([run], start_us=1100, end_us=1110, max_events=3)
        self.assertEqual(
            [(b.get_x(), b.get_width()) for b in ax.patches],
            [(1098, 2), (1099, 2), (1100, 2)],
        )
        self.assertEqual(ax.get_xlim(), (1100, 1110))
        self.assertIn("displayed 3 / 13", ax.get_title())
        self.assertIn("1200 total", ax.get_title())
        self.assertIn("TRUNCATED", ax.get_title())

    def test_large_plot_omits_event_labels(self):
        run = self.run_dir([self.row(i, i + 1, event_id=i) for i in range(60)])
        ax = self.capture([run])
        self.assertEqual(len(ax.patches), 60)
        self.assertFalse(ax.texts)

    def test_zero_duration_is_visible(self):
        ax = self.capture([self.run_dir([self.row(0, 0)])])
        self.assertEqual(ax.patches[0].get_width(), 0)
        self.assertEqual(len(ax.lines), 1)
        self.assertLess(*ax.get_xlim())

    def test_numeric_and_order_validation(self):
        cases = [
            {"start_us": "nan"},
            {"end_us": "inf"},
            {"queue_wait_us": -1},
            {"execution_us": "abc"},
            {"clock_period_ns": "-inf"},
            {"rank": -1},
            {"event_id": "1.5"},
            {"ready_us": 11},
            {"end_us": 9},
            {"ready_cycle": 10001},
            {"end_cycle": 9999},
            {"event_type": "OTHER"},
            {"phase": "unknown"},
        ]
        for updates in cases:
            with self.subTest(updates=updates):
                run = self.run_dir([self.row(**updates)])
                with self.assertRaisesRegex(ValueError, r"event_timing.csv: row 2:"):
                    save_gantt([run], self.output)

    def test_missing_schema_and_ragged_rows(self):
        run = self.run_dir([], fields=("rank", "start_us"))
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            save_gantt([run], self.output)
        self.run_dir([self.row()])
        with (run / "event_timing.csv").open("a") as stream:
            stream.write("0,1\n")
        with self.assertRaisesRegex(ValueError, "incorrect number of columns"):
            save_gantt([run], self.output)

    def test_excluded_and_capped_rows_are_still_validated(self):
        run = self.run_dir([self.row(), self.row(100, 101, execution_us="nan")])
        with self.assertRaisesRegex(ValueError, "execution_us"):
            save_gantt([run], self.output, end_us=25, max_events=1)

    def test_invalid_options(self):
        run = self.run_dir([self.row()])
        cases = [
            {"start_us": -1},
            {"end_us": float("nan")},
            {"start_us": float("inf")},
            {"start_us": 10, "end_us": 10},
            {"start_us": 11, "end_us": 10},
            {"max_events": 0},
            {"max_events": -1},
            {"max_events": 1.5},
            {"max_events": True},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                save_gantt([run], self.output, **kwargs)
        with self.assertRaisesRegex(ValueError, "at least one"):
            save_gantt([], self.output)

    def test_empty_or_nonintersecting_trace(self):
        run = self.run_dir([])
        with self.assertRaisesRegex(ValueError, "no events"):
            save_gantt([run], self.output)
        self.run_dir([self.row()])
        with self.assertRaisesRegex(ValueError, "no events"):
            save_gantt([run], self.output, start_us=100)

    @unittest.skipIf(plt is None, "matplotlib is not installed")
    def test_cli(self):
        run = self.run_dir([self.row()])
        script = Path(__file__).resolve().parents[1] / "plot_timeline.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(run),
                "--output",
                str(self.output),
                "--start-us",
                "5",
                "--end-us",
                "25",
                "--max-events",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.exists())


if __name__ == "__main__":
    unittest.main()
