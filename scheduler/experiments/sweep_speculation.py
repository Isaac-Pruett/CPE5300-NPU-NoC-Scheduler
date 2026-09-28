from pathlib import Path
import csv


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "speculation_sweep.csv"


# Sweep parameters
DRAFT_LENGTHS = [1, 2, 4, 8]
ACCEPTANCE_RATES = [0.25, 0.50, 0.75, 1.00]
BANDWIDTHS_GBPS = [25, 50, 100, 200]


# Synthetic timing model
BASELINE_TOKEN_TIME_MS = 2.0
DRAFT_TOKEN_TIME_MS = 0.20
VERIFY_BASE_TIME_MS = 0.80

# Approximate communication payload per speculative iteration
TRANSFER_MB = 16.0


def network_time_ms(transfer_mb, bandwidth_gbps):
    """
    Convert transfer size and network bandwidth into milliseconds.

    bandwidth_gbps is treated here as GB/s.
    """
    transfer_gb = transfer_mb / 1024.0
    seconds = transfer_gb / bandwidth_gbps
    return seconds * 1000.0


def run_configuration(k, acceptance_rate, bandwidth_gbps):
    accepted_tokens = max(k * acceptance_rate, 0.01)

    baseline_ms = accepted_tokens * BASELINE_TOKEN_TIME_MS

    draft_ms = k * DRAFT_TOKEN_TIME_MS

    # Simple synthetic verification approximation.
    verify_ms = VERIFY_BASE_TIME_MS + 0.10 * k

    comm_ms = network_time_ms(
        transfer_mb=TRANSFER_MB,
        bandwidth_gbps=bandwidth_gbps,
    )

    speculative_ms = draft_ms + verify_ms + comm_ms

    speedup = baseline_ms / speculative_ms

    return {
        "k": k,
        "acceptance_rate": acceptance_rate,
        "bandwidth_gbps": bandwidth_gbps,
        "accepted_tokens": accepted_tokens,
        "baseline_ms": baseline_ms,
        "draft_ms": draft_ms,
        "verify_ms": verify_ms,
        "network_ms": comm_ms,
        "speculative_ms": speculative_ms,
        "speedup": speedup,
    }


def main():
    rows = []

    for k in DRAFT_LENGTHS:
        for acceptance_rate in ACCEPTANCE_RATES:
            for bandwidth in BANDWIDTHS_GBPS:
                result = run_configuration(
                    k=k,
                    acceptance_rate=acceptance_rate,
                    bandwidth_gbps=bandwidth,
                )
                rows.append(result)

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "acceptance_rate",
                "bandwidth_gbps",
                "accepted_tokens",
                "baseline_ms",
                "draft_ms",
                "verify_ms",
                "network_ms",
                "speculative_ms",
                "speedup",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda x: x["speedup"])

    print()
    print("Speculative decoding sweep complete")
    print("-----------------------------------")
    print(f"Configurations tested: {len(rows)}")
    print(f"Results written to:    {OUTPUT_FILE}")
    print()
    print("Best synthetic configuration")
    print("----------------------------")
    print(f"Draft length K:       {best['k']}")
    print(f"Acceptance rate:      {best['acceptance_rate']:.2f}")
    print(f"Bandwidth:            {best['bandwidth_gbps']} GB/s")
    print(f"Baseline time:        {best['baseline_ms']:.3f} ms")
    print(f"Speculative time:     {best['speculative_ms']:.3f} ms")
    print(f"Predicted speedup:    {best['speedup']:.3f}x")
    print()
    print("NOTE: Timing values are synthetic analytical estimates.")
    print("The next step is replacing them with ASTRA-sim measurements.")


if __name__ == "__main__":
    main()
