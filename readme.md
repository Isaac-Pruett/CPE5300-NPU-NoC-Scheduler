# AI prefill/decode NoC simulation experiments

The project runs its Python workloads, Chakra trace conversion, ASTRA-sim build
and simulation, PyTorch profiling, and plotting **inside Docker**. Only Docker
Engine/Desktop with Compose is required on the host. Make is an optional shortcut.
Source and experiment outputs stay in this repository so they can be edited and
inspected normally; no host Python environment or compiler is needed.

## Quick start

Start Docker Desktop, then run from this repository:

```bash
docker compose build
docker compose run --rm scheduler
```

The default command generates a synthetic two-rank prefill/decode pipeline and
runs the congestion-aware analytical backend. Each run creates a new directory
under `scheduler/results/raw/` containing the event IR, two Chakra traces, copies
of the simulator configurations, `simulator.log`, and `summary.json`. Completion
requires both ranks to finish. Runs have a 120-second simulator timeout.

With Make, the equivalent is `make build simulate`. On Linux, Make also passes
your UID/GID to avoid output ownership problems. For direct Compose commands on
Linux, first run `export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)` if your IDs differ
from 1000. Docker Desktop handles the host bind mount on macOS. The image builds
natively on ARM64 or AMD64; no forced x86 emulation is configured.

## Workflows

| Command | Purpose |
| --- | --- |
| `make simulate` | Synthetic prefill/decode pipeline through congestion-aware ASTRA-sim |
| `make speculative` | Existing draft/verify example through congestion-unaware ASTRA-sim |
| `make test` | Event graph, Chakra encoding, and completion-check tests |
| `make profile` | CPU PyTorch execution and Kineto traces |
| `make inspect` | Inspect the generated Kineto trace |
| `make sweep plot` | Existing synthetic speculation estimate sweep and PNG plot |
| `make shell` | Interactive shell with the same tools and mounted source |

See [scheduler/README.md](scheduler/README.md) for direct Docker commands,
workload parameters, output interpretation, and limitations.

See [NEXT_STEPS.md](NEXT_STEPS.md) for the implementation roadmap, baseline
validation results, milestone completion checks, and GitHub handoff steps.

## Container build and source ownership

`Dockerfile` pins ASTRA-sim to
`4ce9ecfe746a3ee5fe47559a11d91644d641fa7f` and fetches its analytical, Chakra,
memory, fmt, and spdlog submodules at their recorded revisions. Both analytical
backends are compiled inside the image. C++ and Python Chakra bindings are
regenerated using the same container-installed `protoc`; the old host-generated
protobuf 7.35.1 binding is not used. Python dependencies are pinned in
`scheduler/requirements.txt` and the Dockerfile (CPU PyTorch).

The historical `astra-sim` gitlink points at an older, incompatible simulator
revision. Its partial files and the sibling `../astra-sim` directory are preserved
but are **not build inputs**. You do not need to initialize that submodule or use
the old manually created containers. The project image fetches its own pinned
source into `/opt/astra-sim`; builds live at `/opt/astra-build`. Changing scheduler
source is immediate through the bind mount; dependency or simulator changes
require `docker compose build` again. The image also includes scheduler source
for standalone use without Compose.

The first build needs internet access and several minutes. `BUILD_JOBS=2 docker
compose build` reduces peak compiler memory. Ubuntu package updates and transitive
Python dependencies are not fully locked, so this is a pinned simulator/toolchain
workflow, not a bit-for-bit image reproducibility guarantee.

## What this establishes for the NoC project

The baseline distinguishes prefill and decode computation, activation payloads,
and autoregressive dependencies. The analytical backend models network transfer
and contention at a high level. **A phase-aware NoC arbitration policy is not yet
implemented.** Phase labels in the traces do not change routing or arbitration.
There are no flits, virtual channels, router buffers, KV-cache traffic model, or
calibrated accelerator timings in this experiment.

Next research steps are a calibrated compute/traffic model, concurrent requests
that contend for links, an implemented phase-aware scheduling policy, and
comparisons against a baseline using latency, throughput, and fairness. A detailed
NoC study additionally needs a suitable packet/flit-level backend. Containerization
and a successful synthetic run do not validate those research claims.

References: [ASTRA-sim documentation](https://astra-sim.github.io/astra-sim-docs/index.html),
[analytical network documentation](https://astra-sim.github.io/astra-network-analytical-docs/index.html),
and [Chakra guide](https://github.com/astra-sim/chakra/blob/0e3cd40c569f0a4cacb6d961bb56be53407abd2f/USER_GUIDE.md).
