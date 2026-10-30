# Next steps: prefill/decode-aware NoC experiments

## Current state

The Docker workflow builds ASTRA-sim, generates synthetic Chakra workloads, and
runs both analytical network backends. CPU profiling, trace inspection, plotting,
and four workload/encoding tests have also been exercised inside Docker.

The current prefill/decode example is one request across two pipeline stages.
Both stages execute both phases. Phase attributes survive trace conversion, but
**no phase-aware network scheduling or router arbitration is implemented yet**.
The existing speculative-decoding example is a separate draft/verify baseline.

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

| Workload / backend | Rank 0 total cycles | Rank 1 total cycles | Compute cycles per rank |
| --- | ---: | ---: | --- |
| Prefill/decode / congestion-aware | 2,824,835 | 2,824,835 | 1,400,000 / 1,400,000 |
| Speculative / congestion-unaware | 901,000 | 901,000 | 400,000 / 500,000 |

These values are regression references for the pinned simulator and current
defaults, not hardware measurements. Raw run directories are ignored by Git.
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

- Add request IDs, token indices, and explicit arrival/release semantics. Merely
  adding an arrival attribute does not delay execution; the simulator must enforce it.
- Generate overlapping requests so prefill and decode transfers actually compete
  for a shared resource. Keep communication tags unique across requests and steps.
- Preserve phase and request metadata through the trace and simulator interfaces.
- Record request arrival and token completion times. Extract time to first token
  (TTFT), time between tokens, request completion time, and tokens per simulated
  second. Define each metric and its time-unit conversion explicitly.
- Extend `scheduler/astra/run.py` and network configurations together if moving
  beyond two ranks; the current runner and generators assume two ranks.
- Add checks for matched communication, autoregressive dependencies, completion
  of every request, and correct metric extraction from a small known timeline.

**Done when:** a deterministic mixed workload demonstrates overlapping traffic,
all requests finish, and per-request/per-token metrics can be checked manually.

## 4. Implement a policy that changes scheduling

Choose the scheduling point explicitly: request admission, endpoint transmission,
or router arbitration. These are different experiments. Identify the queue and
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

Build a new ASTRA-sim experiment driver; do not relabel the existing
`experiments/sweep_speculation.py` formula estimates as simulation measurements.

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
