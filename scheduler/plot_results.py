from pathlib import Path
import csv
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "results" / "speculation_sweep.csv"
OUTPUT_FILE = ROOT / "results" / "speculative_speedup.png"


def load_results():
    rows = []

    with INPUT_FILE.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "k": int(row["k"]),
                    "acceptance_rate": float(row["acceptance_rate"]),
                    "bandwidth_gbps": float(row["bandwidth_gbps"]),
                    "speedup": float(row["speedup"]),
                }
            )

    return rows


def main():
    rows = load_results()

    # Start with a single acceptance rate for a clean plot.
    selected_acceptance = 0.75

    k_values = sorted(set(row["k"] for row in rows))

    plt.figure(figsize=(8, 5))

    for k in k_values:
        filtered = [
            row
            for row in rows
            if row["k"] == k
            and abs(row["acceptance_rate"] - selected_acceptance) < 1e-9
        ]

        filtered.sort(key=lambda x: x["bandwidth_gbps"])

        bandwidth = [row["bandwidth_gbps"] for row in filtered]
        speedup = [row["speedup"] for row in filtered]

        plt.plot(
            bandwidth,
            speedup,
            marker="o",
            label=f"K = {k}",
        )

    plt.axhline(1.0, linestyle="--", linewidth=1)

    plt.xlabel("Network Bandwidth (GB/s)")
    plt.ylabel("Predicted Speedup")
    plt.title(
        f"Speculative Decoding Speedup vs Network Bandwidth\n"
        f"Acceptance Rate = {selected_acceptance:.2f}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=200)

    print(f"Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
