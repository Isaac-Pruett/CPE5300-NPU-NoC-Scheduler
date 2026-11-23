# Next steps: prefill/decode-aware NoC experiments

## Current state

The Docker workflow builds ASTRA-sim, generates synthetic Chakra workloads, and
runs both analytical network backends. CPU profiling, trace inspection, plotting,
and event-timing/Gantt export are available inside Docker. Arrival support is validated by 115 passing tests, including release enforcement
and idle gaps on both analytical backends.

The current prefill/decode example is one request across two pipeline stages.
Both stages execute both phases. Phase attributes survive trace conversion, but
**no phase-aware network scheduling or router arbitration is implemented yet**.
The existing speculative-decoding example is a separate draft/verify baseline.
Offline non-preemptive rank-local compute ordering with discovered FIFO/SJF/LJF
policies is implemented, now with enforced synthetic single-processor releases.
The single-rank compute planner retains one dequeuing policy, choosing only
arrived, dependency-ready jobs at completion decisions without preemption. It
plans offline with known fixed durations, not live Python feedback. Other network
DAGs retain the old adapter, reject nonzero arrivals, and do not schedule traffic.

Docker applies `patches/astra-event-timing.patch` and `patches/astra-arrivals.patch`
alongside the singleton patch; the revision suffix is
`+analytical-single-endpoint-v1+event-timing-v1+arrivals-v1`.
Rebuild with `docker compose build` for arrivals support. `Event.arrival_us` is an
integer microsecond release, default 0, optionally serialized and emitted as a
Chakra uint64 `arrival_us` attribute. Simulator gating/wakeups enforce it without
fake GPU work. Zero-arrival defaults preserve the original results.

Only `single-processor-demo` accepts `--arrival-interval-us N` (default 0, releases
`i * N`) or mutually exclusive `--arrival-times-us` (one per job, unsorted allowed).
Both work with `--jobs`/`--seed`; H100/network-workload arrival options are rejected.

```bash
make benchmark EXPERIMENT=jetson BENCHMARK_ARGS="-- --jobs 1000 --seed 42 --arrival-interval-us 500"
make benchmark EXPERIMENT=jetson BENCHMARK_ARGS="-- --arrival-times-us 500 2000 2000"
```

The runner opts in with `ASTRA_EVENT_TIMING_DIR`, collecting per-rank raw logs of
first observed arrival-satisfied, dependency-free readiness (including policy
edges), dispatch, and
completion callbacks, with ticks and `clock_period_ns` (currently 1 ns). It requires
all three observations for every event and validates bounds, dependencies, and
monotonicity without inferring timestamps.

Each new run exports `event_timing.csv`, sorted transition groups in
`timesteps.json`, and `gantt.png`; benchmark batches also save a combined Gantt and
its manifest path. CSV schema version 2 adds `arrival_us`/`arrival_cycle`.
Timesteps include `arrived` from declared `workload_release_times`, while
ready/start/end remain simulator callbacks. These are postprocessed transitions,
not interactive stepping. Charts use microseconds and rank/type lanes with overlapping RECV
sublanes; SEND end is not assumed delivery, and RECV includes waiting. The default
500-event cap is global to the combined chart with explicit truncation annotations;
`make gantt` supports time-window/limit overrides (see the workflow guide).
Old runs without telemetry must be rerun, not reconstructed.

Single-processor callback metrics (`simulated_single_processor_metrics`, source
`astra_sim_callbacks`, declared releases) coexist with retained analytical
`single_processor_metrics`. `mean_flow_us` is completion minus arrival;
`mean_response_us` and non-preemptive waiting are start minus arrival.
`mean_completion_us` remains absolute for compatibility. Reports/charts prefer
mean flow when available, never mixing it with legacy absolute completion. Comparison latency charts select actual callbacks for
all-supported runs or labeled analytical fallback for legacy-only runs, never
mixed sources. Event observations do not yet provide request/token metrics.

These are synthetic experiments. The PyTorch encoder profiler is not yet a source
of calibrated prefill/decode timings, and the estimate sweep is not an ASTRA-sim
experiment. See [the workflow guide](scheduler/README.md) for commands and limits.

## 1. Reproduce and preserve the baseline

From the repository root, with Docker running:

```bash
docker compose build
make test simulate speculative
```

Inspect the newly created directories under `scheduler/results/raw/`. Each run
must have `status: completed` in `summary.json` and timing records for both ranks.
The validated default configurations on October 30, 2026 produced:

| Workload / backend                | Rank 0 total cycles | Rank 1 total cycles | Compute cycles per rank |
| --------------------------------- | ------------------: | ------------------: | ----------------------- |
| Prefill/decode / congestion-aware |           2,824,835 |           2,824,835 | 1,400,000 / 1,400,000   |
| Speculative / congestion-unaware  |             901,000 |             901,000 | 400,000 / 500,000       |

These historical values are regression references for the then-pinned simulator
and defaults, not new event-timing measurements or hardware measurements. Raw run
directories are ignored by Git.
Preserve a small reviewed summary and the exact parameters when reporting results.

**Done when:** a fresh image build passes the tests and both simulations, without
host Python packages, a host ASTRA-sim checkout, or manually copied traces.

## 2. Define the system and experiment question

Write a short experiment specification before expanding the model:

- Decide what each rank represents: an on-chip compute tile, a pipeline stage,
  or a separate accelerator. The current implementation uses pipeline stages.
- Decide whether both phases share the same compute resources or use separate
  prefill/decode resources. Disaggregation requires a different workload graph
  and an explicit state-transfer model; it is not what the current graph models.
- Define the topology, traffic types, compute assumptions, and request arrivals.
- State the hypothesis, for example: prioritizing decode traffic reduces decode
  tail latency under mixed load while keeping prefill waiting time bounded.
- Decide whether the first study needs message-level scheduling or detailed
  router behavior. Analytical simulation alone cannot support claims about flit
  arbitration, virtual channels, router buffers, or cycle-accurate NoC behavior.

**Done when:** the modeled system, baseline policy, proposed policy, metrics, and
limits are written down so the experiment can be interpreted consistently.

## 3. Add request identity, timing metrics, and concurrent traffic

Start with `scheduler/workloads/prefill_decode.py`,
`scheduler/speculative/event_model.py`, and
`scheduler/tracegen/spec_to_chakra.py`.

- Add request IDs and token indices, and extend the implemented single-processor
  release semantics to multi-request network workloads. Simulator arrival gating
  exists, but the network-DAG planner still rejects nonzero arrivals.
- Generate overlapping requests so prefill and decode transfers actually compete
  for a shared resource. Keep communication tags unique across requests and steps.
- Preserve phase and request metadata through the trace and simulator interfaces.
- Build request/token aggregation on the now-available event timestamps; request
  IDs, token indices, multi-request network arrivals, and token metrics remain undone.
  Record request arrival and token completion times. Extract time to first token
  (TTFT), time between tokens, request completion time, and tokens per simulated
  second. Define each metric and its time-unit conversion explicitly.
- Extend `scheduler/astra/run.py` and network configurations together if moving
  beyond the current single-endpoint compute and two-stage network workloads.
- Add checks for matched communication, autoregressive dependencies, completion
  of every request, and correct metric extraction from a small known timeline.

**Done when:** a deterministic mixed workload demonstrates overlapping traffic,
all requests finish, and per-request/per-token metrics can be checked manually.

## 4. Implement a policy that changes scheduling

Offline compute ordering is current functionality, but phase-aware network policy
is still unimplemented. Choose its scheduling point explicitly: request admission,
endpoint transmission, or router arbitration. These are different experiments. Identify the queue and
resource allocation code that the chosen policy will change.

Implement an explicit phase-unaware baseline (such as FIFO at that queue) and
one phase-aware policy. A candidate policy is decode priority with aging or a
bounded service quota for prefill to prevent starvation. Document its parameters.

The current `phase` and `comm_priority` trace attributes do not implement this.
The chosen scheduling code must receive the relevant metadata and use it when
selecting work. If simulator changes are needed, keep their source or patches in
version control and apply them during the Docker build; do not rely on edits made
inside a disposable container. A detailed NoC backend, if selected, also needs a
pinned container build and integration tests.

**Done when:** a controlled contention test shows the intended service order,
baseline and phase-aware modes differ for the intended reason, and sustained
decode load does not prevent prefill requests from completing.

## 5. Calibrate workloads and compare policies fairly

Replace or parameterize placeholder compute times and payload sizes using a
documented model or measurements. Include prompt length, output length, batch
size, model partitioning, and any modeled KV-cache/state traffic. Record the
source and units of every parameter. Keep synthetic and measured inputs distinct.

Run the same workload, arrivals, topology, and compute model under both policies.
Sweep offered load, prompt/output lengths, traffic mix, bandwidth, and latency.
For randomized arrivals, record seeds and repeat runs. Compare TTFT, decode
latency distributions, throughput, and prefill waiting time/fairness. Only report
link utilization or queue occupancy if the simulator actually records them.

The all-policy batch driver supports single-processor releases. Extend it for
multi-request network arrivals and phase-aware network experiments; those
capabilities and calibration remain undone.
Do not relabel `experiments/sweep_speculation.py` formula estimates as simulation
measurements.

**Done when:** results are reproducible, include benefits and regressions, and
support only claims justified by the selected model and backend.

## GitHub handoff

At the time this document was added, the Docker/workload/documentation changes
were local and uncommitted; they had not been pushed. The configured repository
is [Isaac-Pruett/CPE5300-NPU-NoC-Scheduler](https://github.com/Isaac-Pruett/CPE5300-NPU-NoC-Scheduler).
This paragraph records the handoff state, not a live GitHub status.

The publishing step is to review the complete diff, include the new untracked
source/configuration files, commit on a branch, push that branch, and open a pull
request. Include the Docker validation results and model limitations in its
description. Exclude virtual environments, build products, and raw run outputs.

**Immediate implementation task after publishing:** complete the experiment
specification in step 2, then add request-level metrics and overlapping traffic
in step 3. Do not claim a phase-aware NoC improvement from the current baseline.
