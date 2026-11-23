"""Release-time workload and analytical metrics, independent of ASTRA/planners."""

import random
import unittest

from speculative.event_model import Event, EventTrace, EventType
from workloads.single_processor import (
    build_single_processor_demo,
    single_processor_metrics,
)

MAX_ARRIVAL_US = (2**64 - 1) // 1000


class ArrivalWorkloadTests(unittest.TestCase):
    def test_default_fixed_jobs_and_zero_arrival_metadata(self):
        trace = build_single_processor_demo()
        self.assertEqual(trace.num_ranks, 1)
        self.assertEqual(
            [event.name for event in trace.events],
            ["long_job", "medium_job", "short_job"],
        )
        self.assertEqual([event.duration_us for event in trace.events], [1000, 100, 10])
        self.assertEqual([event.arrival_us for event in trace.events], [0, 0, 0])
        self.assertEqual(trace.parameters["job_arrivals_us"], [0, 0, 0])
        self.assertEqual(trace.parameters["arrival_model"], "all_at_zero")
        self.assertEqual(trace.parameters["arrival_us"], 0)
        self.assertTrue(all(not event.dependencies for event in trace.events))
        self.assertNotIn("compute_schedule", trace.parameters)

    def test_interval_and_explicit_unsorted_arrivals(self):
        for kwargs, expected, model in (
            ({"arrival_interval_us": 25}, [0, 25, 50], "fixed_interval"),
            ({"arrival_times_us": [40, 0, 10]}, [40, 0, 10], "explicit"),
            ({"arrival_times_us": [0, 0, 0]}, [0, 0, 0], "explicit"),
        ):
            with self.subTest(kwargs=kwargs):
                trace = build_single_processor_demo(**kwargs)
                self.assertEqual([event.arrival_us for event in trace.events], expected)
                self.assertEqual(trace.parameters["job_arrivals_us"], expected)
                self.assertEqual(trace.parameters["arrival_model"], model)
                self.assertEqual("arrival_us" in trace.parameters, not any(expected))
                self.assertTrue(all(not event.dependencies for event in trace.events))
                self.assertNotIn("compute_schedule", trace.parameters)
                self.assertEqual(
                    trace.to_dict(), build_single_processor_demo(**kwargs).to_dict()
                )

    def test_generated_durations_and_names_preserve_seeded_sequence(self):
        rng = random.Random(42)
        expected = [rng.randint(10, 1000) for _ in range(8)]
        for kwargs in (
            {},
            {"arrival_interval_us": 12},
            {"arrival_times_us": [8, 0, 6, 2, 4, 5, 1, 3]},
        ):
            with self.subTest(kwargs=kwargs):
                trace = build_single_processor_demo(jobs=8, seed=42, **kwargs)
                self.assertEqual(
                    [event.duration_us for event in trace.events], expected
                )
                self.assertEqual(
                    [event.name for event in trace.events],
                    [f"job_{index}" for index in range(8)],
                )
                self.assertEqual(trace.parameters["seed"], 42)
                self.assertEqual(
                    trace.to_dict(),
                    build_single_processor_demo(jobs=8, seed=42, **kwargs).to_dict(),
                )
        self.assertNotEqual(
            expected,
            build_single_processor_demo(jobs=8, seed=43).parameters["job_durations_us"],
        )

    def test_explicit_arrivals_are_copied(self):
        arrivals = [3, 1, 2]
        trace = build_single_processor_demo(arrival_times_us=arrivals)
        arrivals[0] = 99
        self.assertEqual(trace.parameters["job_arrivals_us"], [3, 1, 2])
        self.assertEqual(trace.events[0].arrival_us, 3)

    def test_invalid_arrival_intervals(self):
        for value in (-1, True, False, 1.0, "1", None, MAX_ARRIVAL_US + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_single_processor_demo(arrival_interval_us=value)

    def test_invalid_explicit_types_counts_and_ranges(self):
        for value in (
            (),
            (0, 1, 2),
            "012",
            0,
            {},
            [],
            [0],
            [0, 1, 2, 3],
            [-1, 0, 0],
            [True, 0, 0],
            [False, 0, 0],
            [1.0, 0, 0],
            ["1", 0, 0],
            [None, 0, 0],
            [MAX_ARRIVAL_US + 1, 0, 0],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_single_processor_demo(arrival_times_us=value)
        with self.assertRaises(ValueError):
            build_single_processor_demo(jobs=2, arrival_times_us=[0, 1, 2])
        with self.assertRaises(ValueError):
            build_single_processor_demo(
                arrival_interval_us=1, arrival_times_us=[0, 1, 2]
            )

    def test_invalid_job_counts(self):
        for jobs in (0, -1, True, False, 1.5, "3"):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                build_single_processor_demo(jobs=jobs)

    def test_arrival_timestamp_boundary_and_interval_product_overflow(self):
        explicit = build_single_processor_demo(arrival_times_us=[MAX_ARRIVAL_US, 0, 1])
        self.assertEqual(explicit.events[0].arrival_us, MAX_ARRIVAL_US)
        interval = build_single_processor_demo(
            jobs=2, arrival_interval_us=MAX_ARRIVAL_US
        )
        self.assertEqual(interval.events[1].arrival_us, MAX_ARRIVAL_US)
        with self.assertRaises(ValueError):
            build_single_processor_demo(jobs=3, arrival_interval_us=MAX_ARRIVAL_US)


class ArrivalMetricsTests(unittest.TestCase):
    @staticmethod
    def scheduled_trace(durations, arrivals, order):
        # A supplied order is authoritative, even when it is not work-conserving.
        return EventTrace(
            num_ranks=1,
            events=[
                Event(
                    index,
                    EventType.COMPUTE,
                    0,
                    f"job_{index}",
                    duration_us=duration,
                    arrival_us=arrival,
                )
                for index, (duration, arrival) in enumerate(zip(durations, arrivals))
            ],
            parameters={
                "workload": "single_processor_demo",
                "compute_schedule": {"0": order},
            },
        )

    def test_default_metrics_remain_backward_compatible(self):
        for order, completion, waiting in (
            ([0, 1, 2], 1070, 700),
            ([2, 1, 0], 410, 40),
        ):
            with self.subTest(order=order):
                trace = self.scheduled_trace([1000, 100, 10], [0, 0, 0], order)
                metrics = single_processor_metrics(trace)
                self.assertEqual(metrics["mean_completion_us"], completion)
                self.assertEqual(metrics["mean_waiting_us"], waiting)
                self.assertEqual(metrics["mean_flow_us"], completion)
                self.assertEqual(metrics["mean_response_us"], waiting)
                self.assertEqual(metrics["makespan_us"], 1110)

    def test_hand_checked_release_times_idle_gaps_and_schedule_order(self):
        trace = self.scheduled_trace([4, 2, 3], [10, 0, 20], [0, 1, 2])
        metrics = single_processor_metrics(trace)
        tasks = metrics["tasks"]
        self.assertEqual([task["arrival_us"] for task in tasks], [10, 0, 20])
        self.assertEqual([task["start_us"] for task in tasks], [10, 14, 20])
        self.assertEqual([task["completion_us"] for task in tasks], [14, 16, 23])
        self.assertEqual([task["waiting_us"] for task in tasks], [0, 14, 0])
        self.assertEqual([task["flow_us"] for task in tasks], [4, 16, 3])
        self.assertEqual([task["response_us"] for task in tasks], [0, 14, 0])
        self.assertEqual(metrics["mean_completion_us"], 53 / 3)
        self.assertEqual(metrics["mean_waiting_us"], 14 / 3)
        self.assertEqual(metrics["mean_flow_us"], 23 / 3)
        self.assertEqual(metrics["mean_response_us"], 14 / 3)
        self.assertEqual(metrics["makespan_us"], 23)
        self.assertEqual(
            metrics["source"], "analytical_from_trace_durations_not_astra_timestamps"
        )
        reordered = single_processor_metrics(
            self.scheduled_trace([4, 2, 3], [10, 0, 20], [1, 0, 2])
        )
        self.assertEqual([task["event_id"] for task in reordered["tasks"]], [1, 0, 2])
        self.assertEqual([task["start_us"] for task in reordered["tasks"]], [0, 10, 20])
        self.assertEqual(reordered["mean_waiting_us"], 0)

    def test_zero_duration_executes_for_one_nanosecond(self):
        metrics = single_processor_metrics(self.scheduled_trace([0, 0], [5, 5], [0, 1]))
        self.assertEqual(metrics["tasks"][0]["duration_us"], 0)
        self.assertAlmostEqual(metrics["tasks"][0]["completion_us"], 5.001)
        self.assertAlmostEqual(metrics["tasks"][1]["start_us"], 5.001)
        self.assertAlmostEqual(metrics["tasks"][1]["waiting_us"], 0.001)
        self.assertAlmostEqual(metrics["makespan_us"], 5.002)
        self.assertAlmostEqual(metrics["mean_flow_us"], 0.0015)
        self.assertAlmostEqual(metrics["mean_response_us"], 0.0005)

    def test_metrics_require_complete_dependency_valid_schedule(self):
        trace = self.scheduled_trace([4, 2, 3], [10, 0, 20], [0, 1, 2])
        for order in ([], [0, 1], [0, 0, 2], [0, 1, 99]):
            trace.parameters["compute_schedule"]["0"] = order
            with (
                self.subTest(order=order),
                self.assertRaisesRegex(ValueError, "complete explicit"),
            ):
                single_processor_metrics(trace)
        trace.parameters["compute_schedule"]["0"] = [1, 0, 2]
        trace.events[1].dependencies = [0]
        with self.assertRaisesRegex(ValueError, "violates"):
            single_processor_metrics(trace)


if __name__ == "__main__":
    unittest.main()
