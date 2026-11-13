"""A policy-choice regression, including actual ASTRA timing when available."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from astra.run import BINARIES, apply_schedule
from schedulers.fifo import FIFO
from schedulers.sjf import ShortestJobFirst
from workloads.policy_demo import build_policy_demo


class PolicyDemoTests(unittest.TestCase):
    def test_short_producer_is_released_before_long_local_work_only_under_sjf(self):
        trace = build_policy_demo()
        fifo = apply_schedule(trace, FIFO)
        sjf = apply_schedule(trace, ShortestJobFirst)
        self.assertEqual(
            fifo.parameters["compute_schedule"], {"0": [0, 1, 7], "1": [4]}
        )
        self.assertEqual(sjf.parameters["compute_schedule"], {"0": [1, 0, 7], "1": [4]})
        # FIFO delays the producer behind the long task; SJF reverses that edge.
        self.assertEqual(fifo.events[1].dependencies, [0])
        self.assertEqual(sjf.events[0].dependencies, [1])
        for scheduled in (fifo, sjf):
            self.assertEqual(scheduled.events[2].dependencies, [1])
            self.assertEqual(scheduled.events[4].dependencies, [3])
            self.assertIn(6, scheduled.events[7].dependencies)
            scheduled.validate()

    @unittest.skipUnless(
        shutil.which(BINARIES["congestion-aware"]), "requires container ASTRA binary"
    )
    def test_sjf_reduces_actual_astra_completion_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            summaries = {}
            for policy in ("fifo", "sjf"):
                output = Path(directory) / policy
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "astra.run",
                        "--workload",
                        "policy-demo",
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
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(set(summary["ranks"]), {"0", "1"})
                summaries[policy] = summary
            completion = {
                policy: max(rank["cycles"] for rank in summary["ranks"].values())
                for policy, summary in summaries.items()
            }
            self.assertLess(completion["sjf"], completion["fifo"], completion)
            for rank in ("0", "1"):
                self.assertEqual(
                    summaries["sjf"]["ranks"][rank]["compute_cycles"],
                    summaries["fifo"]["ranks"][rank]["compute_cycles"],
                )


if __name__ == "__main__":
    unittest.main()
