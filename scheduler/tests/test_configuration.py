"""Configuration failures, reproducible snapshots, and actual backend runs."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from astra.configuration import CONFIG_ROOT, available_configs, load_config
from astra.run import BINARIES

SCHEDULER_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments, cwd=SCHEDULER_ROOT):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SCHEDULER_ROOT), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-m", "astra.run", *map(str, arguments)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=45,
    )


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "custom hardware"
        shutil.copytree(CONFIG_ROOT / "two_rank", self.config)

    def change(self, filename, **updates):
        path = self.config / filename
        data = yaml.safe_load(path.read_text())
        data.update(updates)
        path.write_text(json.dumps(data))  # JSON is also valid YAML.

    def test_bundled_presets_are_complete_and_compatible(self):
        names = {path.name for path in available_configs()}
        self.assertTrue(
            {
                "two_rank",
                "h100_sxm_pair",
                "tpu_v3_pair",
                "ethernet_100g_pair",
                "jetson_orin_nano",
                "jalapeno_experimental",
            }
            <= names
        )
        self.assertNotIn("workloads", names)
        for name in names:
            for backend in BINARIES:
                with self.subTest(name=name, backend=backend):
                    expected_ranks = 1 if name == "jetson_orin_nano" else 2
                    config = load_config(
                        name, num_ranks=expected_ranks, backend=backend
                    )
                    self.assertEqual(config.directory, CONFIG_ROOT / name)
                    self.assertTrue(config.description)

    def test_jetson_has_one_endpoint_and_reference_memory_values(self):
        config = load_config("jetson_orin_nano", num_ranks=1)
        files = dict(config.files)
        network = yaml.safe_load(files["network.yml"])
        system = json.loads(files["system.json"])
        profile = json.loads(files["profile.json"])
        self.assertEqual(network["npus_count"], [1])
        self.assertEqual(network["latency"], [0.0])
        self.assertIn("unused", files["network.yml"].decode())
        with self.assertRaisesRegex(ValueError, "1 ranks but workload requires 2"):
            load_config("jetson_orin_nano", num_ranks=2)
        self.assertEqual(system["local-mem-bw"], 102)
        self.assertEqual(system["peak-perf"], 2.1)
        self.assertEqual(system["roofline-enabled"], 0)
        self.assertEqual(profile["reference_specs"]["memory_capacity_gb"], 8)
        self.assertTrue(profile["sources"])
        self.assertIn("Super", config.description)

    def test_listing_needs_no_simulator(self):
        from astra.run import main

        with (
            patch("sys.argv", ["astra.run", "--list-configs"]),
            patch(
                "astra.run.shutil.which",
                side_effect=AssertionError("must not locate binary"),
            ),
            patch("builtins.print") as output,
        ):
            main()
        self.assertTrue(
            any("h100_sxm_pair" in str(call) for call in output.call_args_list)
        )

    def test_explicit_relative_and_absolute_paths_work_outside_project(self):
        for value in (self.config, Path("custom hardware")):
            with self.subTest(value=value):
                result = run_cli(
                    "--config-dir",
                    value,
                    "--output-dir",
                    self.root / "unused",
                    cwd=self.root,
                )
                # Invalidating the config below must fail before looking for a binary.
                # Here a valid config can either run or report the absent binary.
                if not shutil.which(BINARIES["congestion-aware"]):
                    self.assertIn("ASTRA-sim binary missing", result.stderr)
                else:
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    shutil.rmtree(self.root / "unused")

    def test_invalid_config_fails_before_creating_output(self):
        self.change("network.yml", npus_count=[8])
        output = self.root / "must-not-exist"
        result = run_cli("--config-dir", self.config, "--output-dir", output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("8 ranks but workload requires 2", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_missing_directory_or_files_have_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, "directory does not exist"):
            load_config(self.root / "missing")
        (self.config / "network.yml").unlink()
        with self.assertRaisesRegex(ValueError, "missing required files: network.yml"):
            load_config(self.config)

    def test_malformed_documents_are_rejected_with_filename(self):
        for filename, content in (
            ("network.yml", "topology: [unterminated"),
            ("system.json", "{not-json}"),
            ("remote_memory.json", "[]"),
            ("profile.json", '{"description": 42}'),
        ):
            with self.subTest(filename=filename):
                path = self.config / filename
                original = path.read_bytes()
                path.write_text(content)
                with self.assertRaisesRegex(ValueError, filename):
                    load_config(self.config)
                path.write_bytes(original)

    def test_invalid_network_values_are_rejected(self):
        cases = (
            ("topology", [], "non-empty list"),
            ("topology", ["All-to-All"], "unsupported topology"),
            ("npus_count", [2, 2], "one entry per"),
            ("npus_count", [True], "positive integers"),
            ("npus_count", [0], "positive integers"),
            ("npus_count", [-1], "positive integers"),
            ("bandwidth", [0], "finite positive"),
            ("bandwidth", [float("inf")], "finite positive"),
            ("bandwidth", ["50"], "finite positive"),
            ("latency", [-1], "finite non-negative"),
            ("latency", [float("nan")], "finite non-negative"),
        )
        path = self.config / "network.yml"
        original = path.read_bytes()
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                path.write_bytes(original)
                self.change("network.yml", **{field: value})
                with self.assertRaisesRegex(ValueError, error):
                    load_config(self.config)

    def test_singleton_dimensions_require_fully_connected(self):
        for topology in ("Ring", "Switch"):
            with self.subTest(topology=topology):
                self.change("network.yml", topology=[topology], npus_count=[1])
                with self.assertRaisesRegex(
                    ValueError, "singleton dimensions require FullyConnected"
                ):
                    load_config(self.config)
        self.change("network.yml", topology=["FullyConnected"], npus_count=[1])
        self.assertEqual(load_config(self.config, num_ranks=1).num_ranks, 1)

    def test_dimension_product_and_backend_specific_ring_restriction(self):
        self.change(
            "network.yml",
            topology=["Switch", "FullyConnected"],
            npus_count=[2, 4],
            bandwidth=[50, 25],
            latency=[0, 500],
        )
        system_path = self.config / "system.json"
        system = json.loads(system_path.read_text())
        for key in list(system):
            if key.endswith("-implementation"):
                system[key] = ["ring", "ring"]
        system_path.write_text(json.dumps(system))
        self.assertEqual(load_config(self.config, num_ranks=8).num_ranks, 8)
        with self.assertRaisesRegex(ValueError, "8 ranks but workload requires 2"):
            load_config(self.config, num_ranks=2)
        shutil.copyfile(CONFIG_ROOT / "two_rank/system.json", system_path)
        self.change(
            "network.yml",
            topology=["Ring"],
            npus_count=[2],
            bandwidth=[50],
            latency=[0],
        )
        with self.assertRaisesRegex(ValueError, "two-node Ring"):
            load_config(self.config, backend="congestion-aware")
        load_config(self.config, backend="congestion-unaware")

    def test_roofline_cannot_silently_override_fixed_compute_durations(self):
        self.change("system.json", **{"roofline-enabled": 1})
        with self.assertRaisesRegex(ValueError, "fixed-duration"):
            load_config(self.config)

    def test_invalid_system_and_memory_settings(self):
        for filename, updates, error in (
            ("system.json", {"scheduling-policy": "unknown"}, "scheduling-policy"),
            (
                "system.json",
                {"collective-optimization": "enabled"},
                "collective-optimization",
            ),
            ("system.json", {"all-reduce-implementation": []}, "one algorithm per"),
            ("system.json", {"endpoint-delay": -1}, "endpoint-delay"),
            ("remote_memory.json", {"memory-type": None}, "memory-type"),
        ):
            with self.subTest(updates=updates):
                path = self.config / filename
                original = path.read_bytes()
                self.change(filename, **updates)
                with self.assertRaisesRegex(ValueError, error):
                    load_config(self.config)
                path.write_bytes(original)

    def test_snapshot_and_hashes_preserve_validated_bytes(self):
        config = load_config(self.config)
        original = (self.config / "network.yml").read_bytes()
        self.change("network.yml", bandwidth=[123])
        output = self.root / "snapshot"
        output.mkdir()
        config.write_to(output)
        self.assertEqual((output / "network.yml").read_bytes(), original)
        for filename, digest in config.summary()["sha256"].items():
            self.assertEqual(
                hashlib.sha256((output / filename).read_bytes()).hexdigest(), digest
            )


class HardwareSimulationTests(unittest.TestCase):
    @unittest.skipUnless(
        all(shutil.which(binary) for binary in BINARIES.values()),
        "requires both container ASTRA binaries",
    )
    def test_every_preset_on_both_backends_and_bandwidth_changes_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend in BINARIES:
                summaries = {}
                for preset in available_configs():
                    with self.subTest(backend=backend, preset=preset.name):
                        output = Path(directory) / backend / preset.name
                        config = load_config(preset, backend=backend)
                        arguments = ["--config-dir", preset.name]
                        if config.num_ranks == 1:
                            arguments += ["--workload", "single-processor-demo"]
                        result = run_cli(
                            *arguments,
                            "--backend",
                            backend,
                            "--decode-tokens",
                            "1",
                            "--output-dir",
                            output,
                            "--timeout",
                            "30",
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        summary = json.loads((output / "summary.json").read_text())
                        self.assertEqual(summary["status"], "completed")
                        self.assertEqual(
                            set(summary["ranks"]),
                            {str(rank) for rank in range(config.num_ranks)},
                        )
                        self.assertEqual(summary["hardware"]["name"], preset.name)
                        for filename, digest in summary["hardware"]["sha256"].items():
                            saved = (output / filename).read_bytes()
                            self.assertEqual(saved, (preset / filename).read_bytes())
                            self.assertEqual(hashlib.sha256(saved).hexdigest(), digest)
                        summaries[preset.name] = summary
                h100 = summaries["h100_sxm_pair"]["ranks"]
                # This uncalibrated preset intentionally inherits baseline timing.
                self.assertEqual(
                    summaries["jalapeno_experimental"]["ranks"],
                    summaries["two_rank"]["ranks"],
                )
                ethernet = summaries["ethernet_100g_pair"]["ranks"]
                self.assertLess(
                    max(rank["cycles"] for rank in h100.values()),
                    max(rank["cycles"] for rank in ethernet.values()),
                )
                for rank in ("0", "1"):
                    self.assertEqual(
                        h100[rank]["compute_cycles"], ethernet[rank]["compute_cycles"]
                    )


if __name__ == "__main__":
    unittest.main()
