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
of the simulator configurations, `simulator.log`, and `summary.json`. New runs also
save per-rank raw `event_timing.rank<N>.log`, validated `event_timing.csv`,
`timesteps.json`, and `gantt.png`. Completion requires both ranks to finish and
ready/start/end observations for every event. Runs have a 120-second simulator timeout.

With Make, the equivalent is `make build simulate`. On Linux, Make also passes
your UID/GID to avoid output ownership problems. Make automatically selects
`docker compose` when available, otherwise `docker-compose`; override with
`COMPOSE=...` if needed. For direct Compose commands on
Linux, first run `export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)` if your IDs differ
from 1000. Docker Desktop handles the host bind mount on macOS. The image builds
natively on ARM64 or AMD64; no forced x86 emulation is configured.

## Workflows

| Command            | Purpose                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------- |
| `make simulate`    | Synthetic prefill/decode pipeline through congestion-aware ASTRA-sim                    |
| `make speculative` | Existing draft/verify example through congestion-unaware ASTRA-sim                      |
| `make test`        | Event graph, encoding, completion, timing, and plotting tests                           |
| `make benchmark`   | All discovered compute policies, comparison, per-policy and combined Gantt charts       |
| `make gantt`       | Plot observed event timelines from existing runs (`GANTT_RUNS`, default `COMPARE_RUNS`) |
| `make profile`     | CPU PyTorch execution and Kineto traces                                                 |
| `make inspect`     | Inspect the generated Kineto trace                                                      |
| `make sweep plot`  | Existing synthetic speculation estimate sweep and PNG plot                              |
| `make shell`       | Interactive shell with the same tools and mounted source                                |

For a named batch and a zoomed FIFO/SJF timeline (paths relative to `scheduler/`):

```bash
make benchmark EXPERIMENT=jetson BENCHMARK_DIR=results/raw/my-batch
make gantt GANTT_RUNS="results/raw/my-batch/fifo results/raw/my-batch/sjf" GANTT_OUTPUT=results/my_gantt.png GANTT_ARGS="--start-us 0 --end-us 500 --max-events 2000"
```

The batch parent must not already exist. It contains per-policy runs, `comparison.png`,
a combined `gantt.png`, and `batch.json` recording the Gantt path. `make gantt`
defaults to `results/gantt.png`; its default 500-event limit spans the entire
combined chart and explicitly annotates truncation.

For synthetic single-processor arrivals (default releases remain zero):

```bash
make benchmark EXPERIMENT=jetson BENCHMARK_ARGS="-- --jobs 1000 --seed 42 --arrival-interval-us 500"
make benchmark EXPERIMENT=jetson BENCHMARK_ARGS="-- --arrival-times-us 500 2000 2000"
```

Only `single-processor-demo` accepts these mutually exclusive options: interval
releases are `i * N`, or supply one explicit time per job (unsorted allowed).
`--jobs`/`--seed` remain supported; explicit times must match the job count.
H100/network-workload arrival options are rejected. The single-rank compute
planner uses one persistent dequeuing policy over arrived, dependency-ready jobs
at completion decisions, without preemption. It plans offline with known fixed
durations, not live Python feedback. Other network DAGs retain the old adapter
and reject nonzero arrivals.

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

Docker applies `patches/analytical-single-endpoint.patch`,
`patches/astra-event-timing.patch`, and `patches/astra-arrivals.patch`; the revision
suffix is `+analytical-single-endpoint-v1+event-timing-v1+arrivals-v1`.
**Rebuild with `docker compose build` for arrivals support.** `Event.arrival_us`
is an integer microsecond release, default 0, optionally serialized; the converter
emits Chakra uint64 `arrival_us`. The simulator gates events and schedules wakeups,
not fake GPU work. Zero-arrival defaults preserve the original results.
Instrumentation is opt-in
through `ASTRA_EVENT_TIMING_DIR`, set automatically by the runner. It records first
observed arrival-satisfied, dependency-free readiness (including policy edges),
dispatch, and completion
callbacks as actual ticks plus `clock_period_ns` (`CLOCK_PERIOD`, currently 1 ns).
The runner validates completeness, bounds, dependencies, and monotonicity; it does
not infer timestamps. Old runs without telemetry must be rerun.

Gantt charts show rank/type lanes with overlapping RECV sublanes on a microsecond
axis. SEND completion is not assumed delivery, and RECV includes waiting.
CSV schema version 2 adds declared `arrival_us` and `arrival_cycle`.
`timesteps.json` is a sorted postprocessed arrived/ready/started/completed transition
export with cycle/time references, not interactive stepping. `arrived` is sourced
from declared `workload_release_times`; ready/start/end are simulator callbacks. Single-processor
summaries now separate callback-based `simulated_single_processor_metrics`
(source `astra_sim_callbacks`, declared workload releases) from retained analytical
`single_processor_metrics`. `mean_flow_us` is completion minus arrival;
`mean_response_us` and non-preemptive waiting are start minus arrival.
`mean_completion_us` remains absolute completion for compatibility. Reports/charts
prefer mean flow when available, never mixing it with legacy absolute completion. Comparisons use callback latency when all runs support
it, explicitly labeled analytical fallback for legacy-only inputs, and no
mixed-source latency panel. These are simulated observations, not hardware measurements.

Arrival support is validated by 115 passing tests, including release enforcement
and idle gaps on both analytical backends.
Historical baseline values remain documented in the workflow and roadmap.

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
