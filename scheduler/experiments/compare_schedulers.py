"""Run every discovered scheduler on the same experiment and plot a comparison."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from astra.configuration import load_config
from astra.run import BINARIES
from compare_results import comparison_warnings, load_run, print_report, save_chart
from plot_timeline import save_gantt
from schedulers.discovery import discover_schedulers

EXPERIMENTS = {
    "jetson": ("single-processor-demo", "jetson_orin_nano"),
    "h100": ("policy-demo", "h100_sxm_pair"),
}
WORKLOADS = ("prefill-decode", "speculative", "policy-demo", "single-processor-demo")
MANAGED_OPTIONS = {
    "--policy",
    "--output-dir",
    "--backend",
    "--config-dir",
    "--workload",
    "--list-policies",
    "--list-configs",
    "--timeout",
    "--help",
    "-h",
}


def run_experiment(
    experiment: str = "jetson",
    *,
    workload: str | None = None,
    config_dir: str | None = None,
    backend: str = "congestion-aware",
    output_dir: Path | None = None,
    timeout: float = 120,
    runner_args: list[str] | None = None,
) -> Path:
    """Run all policies sequentially, failing before comparison if any run fails."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    extra = list(runner_args or [])
    for argument in extra:
        option = argument.split("=", 1)[0]
        if option in MANAGED_OPTIONS or (
            option.startswith("--")
            and any(managed.startswith(option) for managed in MANAGED_OPTIONS)
        ):
            raise ValueError(
                f"{argument}: managed experiment options cannot be forwarded after --"
            )
    default_workload, default_config = EXPERIMENTS[experiment]
    workload = workload or default_workload
    config_dir = config_dir or default_config
    policies = discover_schedulers()
    if not policies:
        raise ValueError("no schedulers discovered")
    names = sorted(policies, key=lambda name: (name != "fifo", name))
    # Validate and snapshot metadata before creating any output directory.
    config = load_config(config_dir, backend=backend)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (
        output_dir or Path("results/raw") / f"{stamp}-{experiment}-all-policies"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "experiment": experiment,
        "workload": workload,
        "backend": backend,
        "hardware": config.summary(),
        "runner_args": extra,
        "timeout_seconds": timeout,
        "policies": names,
        "completed_policies": [],
        "runs": {name: str(output / name) for name in names},
        "chart": str(output / "comparison.png"),
        "gantt": str(output / "gantt.png"),
    }
    manifest_path = output / "batch.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    print(
        f"Running policies: {', '.join(names)}; batch artifacts: {output}", flush=True
    )
    try:
        # All children use the same validated config snapshot, even if the source
        # is edited while this batch runs. Individual runs save their own copies.
        snapshot = output / "_config" / config.directory.name
        snapshot.mkdir(parents=True)
        config.write_to(snapshot)
        runs = []
        for name in names:
            manifest["current_policy"] = name
            save_manifest()
            command = [
                sys.executable,
                "-m",
                "astra.run",
                "--workload",
                workload,
                "--config-dir",
                str(snapshot),
                "--backend",
                backend,
                "--policy",
                name,
                "--output-dir",
                str(output / name),
                "--timeout",
                str(timeout),
                *extra,
            ]
            result = subprocess.run(command, check=False, timeout=timeout + 15)
            if result.returncode:
                raise RuntimeError(
                    f"policy {name!r} failed (exit {result.returncode}); "
                    f"inspect {output / name}; no comparison chart generated"
                )
            run = load_run(output / name)
            if run.summary.get("scheduling_policy") != name:
                raise ValueError(
                    f"policy {name!r}: summary does not match requested scheduler"
                )
            runs.append(run)
            manifest["completed_policies"].append(name)
            save_manifest()
        warnings = comparison_warnings(runs)
        print_report(runs, warnings)
        save_chart(runs, output / "comparison.png", warnings)
        save_gantt([run.directory for run in runs], output / "gantt.png")
        manifest["comparison_warnings"] = warnings
        manifest["status"] = "completed"
        manifest.pop("current_policy", None)
        print(f"\nComparison chart: {output / 'comparison.png'}", flush=True)
        print(f"Gantt chart: {output / 'gantt.png'}", flush=True)
        return output
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        save_manifest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="jetson")
    parser.add_argument(
        "--workload", choices=WORKLOADS, help="Override the experiment workload"
    )
    parser.add_argument(
        "--config-dir", help="Override the experiment hardware preset/directory"
    )
    parser.add_argument("--backend", choices=BINARIES, default="congestion-aware")
    parser.add_argument(
        "--output-dir", type=Path, help="New batch directory; default is timestamped"
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Per-policy ASTRA timeout in seconds"
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Shared workload arguments, following -- (e.g. -- --decode-tokens 8)",
    )
    args = parser.parse_args()
    extra = args.runner_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        run_experiment(
            args.experiment,
            workload=args.workload,
            config_dir=args.config_dir,
            backend=args.backend,
            output_dir=args.output_dir,
            timeout=args.timeout,
            runner_args=extra,
        )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
