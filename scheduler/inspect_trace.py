from pathlib import Path
import json
import sys


def load_trace(path):
    with path.open() as f:
        return json.load(f)


def inspect_kineto_trace(data):
    events = data.get("traceEvents", [])

    total_events = len(events)

    cpu_events = 0
    cuda_events = 0
    duration_us = 0.0

    event_names = {}

    for event in events:
        name = event.get("name", "unknown")
        category = str(event.get("cat", "")).lower()

        duration_us += float(event.get("dur", 0.0))

        event_names[name] = event_names.get(name, 0) + 1

        if "cuda" in category or "gpu" in category:
            cuda_events += 1

        if "cpu" in category:
            cpu_events += 1

    print()
    print("PyTorch / Kineto Trace Summary")
    print("------------------------------")
    print(f"Total events:        {total_events}")
    print(f"CPU events:          {cpu_events}")
    print(f"CUDA/GPU events:     {cuda_events}")
    print(f"Summed duration:     {duration_us / 1000:.3f} ms")

    print()
    print("Most common operations")
    print("----------------------")

    sorted_events = sorted(
        event_names.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for name, count in sorted_events[:15]:
        print(f"{count:6d}  {name}")


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python inspect_trace.py traces/kineto_trace_rank0.json")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Trace not found: {path}")
        sys.exit(1)

    data = load_trace(path)

    if isinstance(data, dict) and "traceEvents" in data:
        inspect_kineto_trace(data)
    else:
        print("Trace loaded, but this does not appear to be a Kineto trace.")
        print(f"Top-level type: {type(data).__name__}")

        if isinstance(data, dict):
            print("Top-level keys:")
            for key in data.keys():
                print(f"  {key}")


if __name__ == "__main__":
    main()
