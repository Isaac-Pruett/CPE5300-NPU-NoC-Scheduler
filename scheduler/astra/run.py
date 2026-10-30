"""Generate a workload, convert it, and run ASTRA-sim in one container."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess

from speculative.speculative_decode import build_speculative_iteration
from tracegen.spec_to_chakra import convert_trace
from workloads.prefill_decode import add_workload_arguments, build_prefill_decode


CONFIG = Path(__file__).resolve().parent / "configs" / "two_rank"
BINARIES = {
    "congestion-aware": "AstraSim_Analytical_Congestion_Aware",
    "congestion-unaware": "AstraSim_Analytical_Congestion_Unaware",
}


def completed_ranks(output: str, num_ranks: int) -> dict[str, dict[str, int]]:
    """Require explicit completion from every rank, not just an exit code of 0."""
    ranks = {}
    for rank, cycles, exposed in re.findall(
        r"sys\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles", output
    ):
        ranks[rank] = {"cycles": int(cycles), "exposed_communication_cycles": int(exposed)}
    if set(ranks) != {str(rank) for rank in range(num_ranks)}:
        raise RuntimeError("ASTRA-sim did not report completion for every rank; inspect simulator.log")
    for rank, cycles in re.findall(r"sys\[(\d+)\], GPU time: (\d+)", output):
        if rank in ranks:
            ranks[rank]["compute_cycles"] = int(cycles)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["prefill-decode", "speculative"], default="prefill-decode")
    parser.add_argument("--backend", choices=BINARIES, default="congestion-aware")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing runs are never overwritten")
    parser.add_argument("--timeout", type=float, default=120, help="Simulator timeout in seconds")
    add_workload_arguments(parser)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--accepted-tokens", type=int, default=3)
    parser.add_argument("--draft-compute-us", type=int, default=100)
    parser.add_argument("--target-verify-us", type=int, default=500)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    if args.workload == "prefill-decode":
        trace = build_prefill_decode(
            args.decode_tokens, args.prefill_compute_us, args.decode_compute_us,
            args.prefill_bytes, args.decode_bytes,
        )
    else:
        trace = build_speculative_iteration(
            args.k, args.accepted_tokens, args.draft_compute_us, args.target_verify_us,
        )
    binary = shutil.which(BINARIES[args.backend])
    if binary is None:
        parser.error("ASTRA-sim binary missing. Run this command through docker compose.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = (args.output_dir or Path("results/raw") / f"{stamp}-{args.workload}-{args.backend}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    trace.write_json(output_dir / "events.json")
    convert_trace(trace, output_dir / "workload")
    for name in ("system.json", "network.yml", "remote_memory.json"):
        shutil.copyfile(CONFIG / name, output_dir / name)

    command = [
        binary,
        f"--workload-configuration={output_dir / 'workload'}",
        f"--system-configuration={output_dir / 'system.json'}",
        f"--network-configuration={output_dir / 'network.yml'}",
        f"--remote-memory-configuration={output_dir / 'remote_memory.json'}",
    ]
    revision_file = Path("/opt/astra-sim-revision")
    summary = {
        "status": "running", "workload": args.workload, "backend": args.backend,
        "timing_source": "synthetic", "parameters": trace.parameters,
        "astra_sim_revision": revision_file.read_text().strip() if revision_file.exists() else "unknown",
        "command": command,
    }
    summary_file = output_dir / "summary.json"

    def save_summary() -> None:
        summary_file.write_text(json.dumps(summary, indent=2) + "\n")

    save_summary()
    log = output_dir / "simulator.log"
    print(f"Running {args.workload} with {args.backend}; artifacts: {output_dir}", flush=True)
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=output_dir, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
        output = log.read_text()
        print(output, end="")
        summary["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"ASTRA-sim exited with {result.returncode}; inspect {log}")
        summary["ranks"] = completed_ranks(output, trace.num_ranks)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        save_summary()
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
