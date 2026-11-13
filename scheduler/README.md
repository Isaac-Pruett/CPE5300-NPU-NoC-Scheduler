# Containerized inference workload experiments

Run the commands below from the **repository root**, after `docker compose build`.
Every Python command and simulator process runs in the project container. Do not
create a host virtual environment or copy traces into a container by ID.

For what to implement next, see the [project roadmap](../NEXT_STEPS.md).

## Prefill/decode simulation

```bash
docker compose run --rm scheduler python -m astra.run \
  --workload prefill-decode --backend congestion-aware \
  --decode-tokens 4 \
  --prefill-compute-us 1000 --decode-compute-us 100 \
  --prefill-bytes 1048576 --decode-bytes 4096
```

This is a **synthetic two-stage pipeline**. Both ranks execute prefill and decode;
the ranks are not separate prefill/decode servers. Each step computes on rank 0,
sends activations to rank 1, computes there, then returns a 4-byte sampled token.
Rank 0 waits for that feedback before starting the next autoregressive step.
The prefill pass produces initial token feedback, followed by four additional
decode passes by default. Compute durations are per rank, in microseconds.
Activation sizes are configurable synthetic byte counts, not measured tensors.

Events carry a `phase` field (`prefill` or `decode`) that survives conversion as a
Chakra attribute. This is descriptive metadata; it does not implement a router
priority policy. The sequential single-request example is a correctness baseline,
not an experiment demonstrating congestion reduction.

`--backend congestion-unaware` runs the same graph without contention modeling.
The runner uses `astra/configs/two_rank/`: two directly connected NPUs, 50 GB/s
bandwidth, 500 ns link latency, fixed compute timing, and no remote memory
expansion. `FullyConnected` supports both analytical backends; a two-node `Ring`
creates duplicate links and aborts in the congestion-aware backend. These are
generic baseline values, not a calibrated on-chip interconnect. Edit those
configuration files to explore link parameters; the topology must continue to
contain two ranks for these workloads.

## Offline compute scheduling

The runner defaults to `--policy fifo`. Use `--policy sjf` for shortest-job-first
or `--policy none` to preserve the original workload dependencies:

```bash
docker compose run --rm scheduler python -m astra.run --policy sjf
```

`astra/run.py` wraps eligible compute events in `Task` objects and asks the selected
Python scheduler for the next event. It adds rank-local compute dependencies before
Chakra conversion, modeling one non-preemptive compute resource per rank. FIFO
selects whole events in admission order, without time slicing. Each selection uses a fresh
policy instance containing the current eligible tasks; persistent policy state is not
carried between selections. The input trace is not mutated.

The adapter traverses dependencies and matched SEND-to-RECV edges symbolically to
avoid ordering work ahead of its prerequisites. This is an offline ordering, not a
simulation of completion times, arrivals, or network congestion. Communication tags
must identify unique pairs, and blocked communication/dependency graphs are rejected.
Communication events themselves are not reordered or assigned priority by this policy.

The chosen policy is recorded in `summary.json` and the trace parameters; parameters
also contain `compute_schedule`, listing compute event IDs in order for each rank.
The existing sequential examples offer little policy choice; independent overlapping
work is needed to demonstrate a difference. This does not implement preemption or
router arbitration.

## Existing speculative-decoding simulation

```bash
docker compose run --rm scheduler python -m astra.run \
  --workload speculative --backend congestion-unaware \
  --k 4 --accepted-tokens 3
```

Rank 0 performs four 100 us draft computations, sends `K * 4` bytes of token IDs,
and receives a 4-byte acceptance result. Rank 1 receives the candidates, performs
500 us of verification, and returns the result. `--draft-compute-us` and
`--target-verify-us` override those timings. `accepted_tokens` records experiment
metadata only: this one-iteration graph does not model rejection, rollback, or
follow-on decoding. The earlier host/container workflow reported 901000 cycles
for each rank, with 400000 and 500000 compute cycles respectively; treat that as
a historical baseline, not a universal result across simulator revisions.

## Saved outputs

Each invocation creates `scheduler/results/raw/<UTC timestamp>-<workload>-<backend>/`:

- `events.json`: validated simulator-independent event graph and parameters.
- `workload.0.et`, `workload.1.et`: Chakra traces, with matched sends/receives.
- `system.json`, `network.yml`, `remote_memory.json`: exact configuration copies.
- `simulator.log` and simulator-generated `log/`: diagnostics and timing output.
- `summary.json`: status, simulator revision, command, parameters, and per-rank
  total, exposed-communication, and compute cycles when reported.

The cycle values are ASTRA-sim's simulated timing, not host elapsed time. The
`GPU time` statistic means modeled accelerator compute, even when Docker runs on
a CPU. Synthetic timing assumptions remain synthetic after simulation. Exposed
communication can include dependency waits; it is not a pure link-utilization
metric. TTFT and per-token latency are not extracted by this runner yet.

Use `--output-dir results/raw/my-experiment` for a named run; it must not already
exist. `--timeout 120` controls the simulator time limit. Failure or timeout leaves
the run artifacts and a failed summary. The runner also rejects an exit-code-zero
run if any rank's completion record is missing.

## A workload that distinguishes FIFO and SJF

```bash
make policy-demo
```

This runs both policies on the synthetic `--workload policy-demo` workload and
saves summaries and a chart under `scheduler/results/raw/policy-demo/`. The two
run directories must not already exist. For another run, choose a new parent:

```bash
make policy-demo POLICY_DEMO_DIR=results/raw/policy-demo-2
```

Rank 0 initially has two independent tasks: a 1000 us local task and a 100 us
producer. The producer sends 4 bytes to unlock a 1000 us consumer on rank 1.
Rank 1 sends feedback, and a 1 us rank-0 join waits for both branches.

- FIFO runs the long local task first, delaying the producer and remote consumer.
- SJF runs the producer first, allowing the remote consumer to overlap the long
  local task. Both policies perform exactly the same computation and communication.

With the pinned congestion-aware backend and default configuration, the validated
completion times were **2,102,000 cycles (FIFO)** and **1,102,000 cycles (SJF)**,
a **1.907x** speedup. This is an intentionally constructed scheduling regression,
not evidence that SJF always wins or that router arbitration has changed.

`tests/test_policy_demo.py` checks the chosen dependencies and, when the ASTRA
binary is available (including `make test` in Docker), runs both simulations and
asserts lower SJF completion cycles with equal compute cycles per rank.

## Compare policy results

After completing two runs (each output directory must be new):

```bash
docker compose run --rm scheduler python -m astra.run \
  --policy fifo --output-dir results/raw/fifo-demo
docker compose run --rm scheduler python -m astra.run \
  --policy sjf --output-dir results/raw/sjf-demo

docker compose run --rm scheduler python compare_results.py \
  results/raw/fifo-demo results/raw/sjf-demo \
  --output results/policy_comparison.png
```

From the repository root, `make compare` runs the comparison for the existing
`fifo-demo` and `sjf-demo` directories; it does not generate new simulations. Override
the inputs, output, or extra comparison options as needed (paths are relative to
`scheduler/` inside the container):

```bash
make compare
make compare COMPARE_RUNS="results/raw/fifo-demo results/raw/sjf-comparison"
make compare COMPARE_OUTPUT=results/my_comparison.png
make compare COMPARE_ARGS=--no-plot
```

The comparison prints overall completion cycles, per-rank timing, and offline
compute orders with event IDs and names. Open `scheduler/results/policy_comparison.png`
in Zed for the chart. Use `--no-plot` for tables only. Input paths can also point
directly to `summary.json`; the script reads `events.json` beside each summary.

The first run is the speedup baseline. Overall completion is the maximum rank
completion cycles, not their sum. Failed/incomplete runs and missing rank records
are rejected. Differences in workload parameters, backend, simulator revision,
rank count, or saved configurations produce warnings and suppress speedup ratios.
These checks flag known confounders, not prove workload equivalence. Identical
completion cycles are expected when the workload offers no scheduling choice.
The chart shows measured simulated completion cycles, not per-event timelines,
TTFT, or host runtime. `plot_results.py` remains the separate speculation-estimate
plotter and is not used for ASTRA run comparisons.

## Generate and inspect traces separately

```bash
docker compose run --rm scheduler python -m workloads.prefill_decode
docker compose run --rm scheduler python -m tracegen.spec_to_chakra \
  --input traces/prefill_decode_events.json --output-prefix traces/prefill_decode

docker compose run --rm scheduler python -m speculative.speculative_decode \
  --k 4 --accepted-tokens 3 --output traces/speculative_events.json
docker compose run --rm scheduler python -m tracegen.spec_to_chakra \
  --input traces/speculative_events.json --output-prefix traces/spec_decode

docker compose run --rm scheduler python -m chakra.src.jsonizer.jsonizer \
  --input_filename traces/prefill_decode.0.et \
  --output_filename traces/prefill_decode.0.json
```

Repeat the last command for rank 1. The bundled jsonizer writes consecutive JSON
objects, not a JSON array. ASTRA-sim accepts a workload **prefix**, without the rank
or `.et` suffix. The converter encodes rank-local ordering in `data_deps` and uses
`comm_src`, `comm_dst`, `comm_size`, `comm_tag`, and `comm_priority` for communication.
`--comm-priority` sets a common trace attribute; the analytical backend does not
use it to implement phase-aware link arbitration.

## Profiling, estimates, plotting, and tests

```bash
docker compose run --rm scheduler python main.py
docker compose run --rm scheduler python profile_workload.py
docker compose run --rm scheduler python inspect_trace.py traces/kineto_trace_rank0.json
docker compose run --rm scheduler python experiments/sweep_speculation.py
docker compose run --rm scheduler python plot_results.py
docker compose run --rm scheduler python -m unittest discover -s tests -v
```

The image includes CPU PyTorch and headless matplotlib. The profiler records the
existing tiny Transformer encoder, not a KV-cached autoregressive LLM. On this
macOS Docker workflow it measures Linux CPU execution, not Apple GPU or CUDA
performance. Its JSON traces are not automatically linked or converted to the
synthetic ASTRA-sim inputs. Full Chakra PyTorch trace-link dependencies are outside
the current workflow; the image includes the schema, framing utilities, and jsonizer
used by the synthetic converter.

`experiments/sweep_speculation.py` produces **synthetic analytical estimates**,
not ASTRA-sim results. It writes `results/speculation_sweep.csv`; plotting writes
`results/speculative_speedup.png`. These commands intentionally replace those
existing outputs. The legacy sweep's `bandwidth_gbps` column is interpreted as
GB/s, not gigabits/s. Profiling writes/replaces files in `traces/`.

To work interactively, use `docker compose run --rm scheduler bash`. To run tests
and both simulation baselines together, use `make test simulate speculative`.
