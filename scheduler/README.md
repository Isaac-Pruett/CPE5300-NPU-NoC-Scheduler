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
generic baseline values, not a calibrated on-chip interconnect. Select another
configuration using `--config-dir`; the topology must contain the same number
of ranks as the workload (currently two).

## Select a hardware configuration

Rebuild the image after this change to install the pinned YAML parser:

```bash
docker compose build
make configs
make simulate CONFIG_DIR=h100_sxm_pair
make speculative CONFIG_DIR=tpu_v3_pair
make simulate CONFIG_DIR=jalapeno_experimental
make jetson-demo
make policy-demo CONFIG_DIR=ethernet_100g_pair POLICY_DEMO_DIR=results/raw/ethernet-demo
```

The runner accepts a bundled preset name or a directory path:

```bash
docker compose run --rm scheduler python -m astra.run \
  --config-dir h100_sxm_pair --policy sjf

docker compose run --rm scheduler python -m astra.run \
  --config-dir astra/configs/ethernet_100g_pair

docker compose run --rm scheduler python -m astra.run --list-configs
```

| Preset                  | Modeled endpoints                                               | Link GB/s per direction | Link latency (ns) |
| ----------------------- | --------------------------------------------------------------- | ----------------------: | ----------------: |
| `two_rank` (default)    | Generic baseline pair                                           |                      50 |               500 |
| `h100_sxm_pair`         | H100 SXM-inspired pair                                          |                     450 |               500 |
| `tpu_v3_pair`           | TPU v3-inspired pair                                            |                      50 |               500 |
| `ethernet_100g_pair`    | Generic 100 Gbit/s pair                                         |                    12.5 |             2,000 |
| `jetson_orin_nano`      | One Jetson Orin Nano 8GB (Super mode)-inspired compute endpoint |           N/A (no link) |               N/A |
| `jalapeno_experimental` | Experimental OpenAI/Broadcom Jalapeño-inspired pair             |                      50 |               500 |

These are **idealized experiments**, not calibrated hardware models. All presets
except Jetson have two endpoints; Jetson has one. They use `FullyConnected` and
work with both analytical backends after applying the bundled singleton patch. The H100 preset
assumes an ideal 450 GB/s per direction to one peer, half the advertised
[900 GB/s aggregate NVLink bandwidth](https://www.nvidia.com/en-eu/data-center/h100/).
It does not model an HGX board or NVSwitch. The TPU preset records
[TPU v3 compute and memory reference values](https://docs.cloud.google.com/tpu/docs/v3),
but uses a synthetic link matching the baseline; it does not model a TPU pod or
its torus. Ethernet bandwidth is ideal line rate without protocol overhead.
All link latencies are synthetic. Each preset's `profile.json` records sources
and assumptions and is copied into the run output.

`jetson_orin_nano` represents **one 8GB Orin Nano device with one serialized
compute resource**, using
[NVIDIA's Super-mode reference specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/):
102 GB/s LPDDR5 bandwidth and 2.1 FP32 TFLOPS. These are not the original lower-clock
15W figures, nor the 4GB variant or Orin Nano 2. The singleton topology has no
inter-device traffic; its positive bandwidth field is an unused parser placeholder,
not an Ethernet connection or the memory bus. The physical Jetson has multiple CPU
cores and GPU execution units; this experiment deliberately models only one
non-preemptive compute resource. Compute remains trace-driven; memory capacity,
shared-memory contention, internal NoC behavior, DVFS, power, and thermals are not
modeled. This is not a prediction of measured Jetson inference performance.

### Single-processor scheduler comparison

```bash
make jetson-demo
# For another run, select a new output parent:
make jetson-demo JETSON_DEMO_DIR=results/raw/jetson-demo-2
```

This runs `--workload single-processor-demo --config-dir jetson_orin_nano` for
FIFO and SJF, then saves the chart to
`scheduler/results/raw/jetson-demo/comparison.png`. The original prefill/decode,
speculative, and policy-demo workloads still require two ranks and are rejected
with this singleton preset. Direct invocation is also supported:

```bash
docker compose run --rm scheduler python -m astra.run \
  --workload single-processor-demo --config-dir jetson_orin_nano --policy sjf
```

The demo has independent 1000, 100, and 10 us compute jobs, all arriving at time
zero. FIFO chooses long-to-short; SJF chooses short-to-long. There are no network
operations, idle gaps, or scheduling/context-switch costs. SJF is given exact job
durations; real workloads may require estimates.

| Metric                   |      FIFO |       SJF | Source                                       |
| ------------------------ | --------: | --------: | -------------------------------------------- |
| Total completion cycles  | 1,110,000 | 1,110,000 | Validated ASTRA output                       |
| Mean job completion (us) |     1,070 |       410 | Analytical from selected order and durations |
| Mean job waiting (us)    |       700 |        40 | Analytical from selected order and durations |
| Makespan (us)            |     1,110 |     1,110 | Analytical sum of durations                  |

The latency values are saved under `single_processor_metrics` in `summary.json`;
per-task start/completion/waiting values there are **analytical, not extracted ASTRA
timestamps or Jetson measurements**. The comparison prints a separate labeled table
and adds a second chart panel for mean completion time. Equal total runtime is
expected: reordering the same work on one busy resource does not reduce its sum.
This demonstrates lower average latency for this workload, not improved overall
throughput, lower scheduler implementation overhead, starvation freedom, or a
hardware performance claim. No per-event latency is invented for other workloads.

Upstream analytical ASTRA rejects `npus_count: [1]` despite FullyConnected topology
construction supporting it. `Dockerfile` applies
`patches/analytical-single-endpoint.patch` to the pinned source before compilation.
Only singleton FullyConnected dimensions are allowed; singleton Ring/Switch remain
invalid. The simulator revision recorded in summaries has the suffix
`+analytical-single-endpoint-v1` so patched runs can be distinguished. Rebuild with
`docker compose build` before running these commands.

`jalapeno_experimental` provides a starting point for experiments inspired by
[OpenAI and Broadcom's Jalapeño](https://openai.com/index/openai-broadcom-jalapeno-inference-chip/).
Each rank represents one abstract chip. It uses the original baseline's network
and system values because the official announcements reviewed do not provide the
interconnect details needed to calibrate this model. Its 50 GB/s bandwidth,
500 ns latency, memory bandwidth, and peak compute settings are **synthetic
placeholders, not Jalapeño specifications**. The profile records this explicitly.
It does not model Jalapeño's internal NoC, KV-cache placement, power, or specialized
kernels. With equal workload inputs it should match the baseline's timing;
reported chip benchmark speedups are not applied as simulation multipliers.
Copy the preset to a custom directory to sweep assumptions or replace them with
documented measurements as they become available.

Compute durations still come from the workload's `--*-compute-us` arguments.
`roofline-enabled` must be `0`: these traces contain durations, not the operation
and memory counts needed to predict hardware compute performance. `peak-perf`
and `local-mem-bw` do not rescale these compute durations. In particular, the
TPU preset and baseline are expected to produce equal timings for equal inputs.

To create a custom profile, copy a complete preset directory and edit it. The
required files are `system.json`, `network.yml`, and `remote_memory.json`.
`profile.json` is optional; when present it must contain a `description` string.
Network YAML must contain equal-length `topology`, `npus_count`, `bandwidth`, and
`latency` lists. Supported topology names are `FullyConnected`, `Ring`, and
`Switch`; the product of `npus_count` must match the workload's rank count.
The runner rejects invalid dimensions, non-finite/negative timings, non-positive
bandwidth, and two-node rings on the congestion-aware backend before creating
an output directory. It checks common system fields; ASTRA remains responsible
for backend-specific optional settings.

Bare bundled names resolve independently of the working directory. Other paths
are relative to the process working directory; use `./two_rank` to explicitly
select a local directory with the same name as a preset. Docker commands run
in `/workspace/scheduler`, with the host `scheduler/` directory mounted there.
Place custom configs under that directory or supply an additional Docker mount.
The incomplete `NVIDIA_H100` and `TPU_v3` draft folders are not listed as presets;
the complete presets above are separate from those drafts.

Every `summary.json` now contains a `hardware` record with the preset/directory
name, source path, rank count, description, and SHA-256 hashes. The exact validated
config bytes are saved with each run. Existing comparisons check saved simulator
config files, so comparisons between different hardware configurations suppress
policy speedup ratios and report the differences.

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

### Add or discover a scheduler

```bash
make policies
# Or directly:
docker compose run --rm scheduler python -m astra.run --list-policies
```

Schedulers are automatically discovered from public `.py` modules directly inside
`scheduler/schedulers/`; there is no central import list or `POLICIES` dictionary
to update. Currently the discovered policies are `fifo`, `sjf`, and `ljf`.

To add a policy, create a module such as `schedulers/my_policy.py` containing a
concrete `Scheduler` subclass with a no-argument constructor (inheriting the base
constructor is fine), `admit(tasks)`, and `next()` returning an admitted task.
The default CLI name is the module filename with underscores replaced by hyphens:
`my_policy.py` becomes `--policy my-policy`. Run a new command to discover it;
Compose mounts the source, so no image rebuild is needed for Python-only changes.

Discovery ignores abstract classes, imported class aliases, private modules whose
names start with `_`, and subpackages. A helper class can set `discoverable = False`
to opt out; that marker applies only to the class declaring it, so subclasses are
still discovered. `PrioHeapOnFIFO` uses this opt-out. `Scheduler` is an actual ABC:
a subclass must implement its abstract `next()` method before it is eligible.

Set `policy_name = "custom-name"` on a class to override its module-derived CLI
name (also needed when multiple policies share one module). Names must use lowercase
letters/digits separated by single hyphens and start with a letter. `none` is
reserved. Duplicate names, required constructor arguments, invalid names, and broken
module imports cause an actionable CLI error rather than silently hiding policies.
Discovery imports modules but does not instantiate policies; only trusted project
code belongs in this package, and modules should avoid import-time side effects.

The adapter still constructs a fresh policy per selection: discovery does not add
persistent queues, runtime congestion feedback, or preemption. Existing Make demos
explicitly compare FIFO/SJF; discovery makes new policies available through `--policy`
but does not automatically add them to every experiment or comparison chart.

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

## Benchmark all discovered schedulers

The all-scheduler batch runner runs every automatically discovered policy on one
experiment, then compares the completed runs:

```bash
make benchmark                     # Defaults to EXPERIMENT=jetson
make benchmark EXPERIMENT=jetson
make benchmark EXPERIMENT=h100
```

| Experiment | Default workload        | Default configuration |
| ---------- | ----------------------- | --------------------- |
| `jetson`   | `single-processor-demo` | `jetson_orin_nano`    |
| `h100`     | `policy-demo`           | `h100_sxm_pair`       |

FIFO runs first and is the comparison baseline when present; all other policies
run in alphabetical order. If FIFO is absent, the first alphabetical policy is
the baseline. The batch never includes `none`. New discoverable policies are
included automatically, unlike the fixed-policy demos above.

By default, each batch creates a new timestamped parent directory at
`scheduler/results/raw/<stamp>-<experiment>-all-policies/`, containing per-policy
run directories, a `batch.json` manifest, and `comparison.png`. To select an
explicit parent, supply a **new, nonexistent** path (relative to `scheduler/`
inside the container):

```bash
make benchmark EXPERIMENT=jetson BENCHMARK_DIR=results/raw/my-batch
```

Existing output parents are rejected; batches never overwrite earlier results.
The batch stops at any failed or incomplete run and does not generate a partial
comparison.

Use `BENCHMARK_ARGS` to override the backend, workload, or configuration:

```bash
make benchmark BENCHMARK_ARGS="--backend congestion-unaware"
make benchmark BENCHMARK_ARGS="--workload prefill-decode --config-dir h100_sxm_pair"
```

Forward workload-specific parameters after `--`:

```bash
make benchmark EXPERIMENT=h100 BENCHMARK_ARGS="--workload prefill-decode -- --decode-tokens 8"
```

For a longer single-Jetson trace, generate 1,000 independent jobs with reproducible
random durations (uniform integer 10–1,000 us):

```bash
make benchmark EXPERIMENT=jetson BENCHMARK_ARGS="-- --jobs 1000 --seed 42"
```

Omitting `--jobs` keeps the fixed three-job regression. The generated admission
order is not sorted, so FIFO normally differs from both SJF and LJF. The seed and
complete duration list are recorded in every run; all policies receive the same
jobs. All jobs still arrive at time zero on one compute resource, with no modeled
switching overhead. A longer trace is not a model of concurrent CPU/GPU execution
or real request arrivals. Mean latency remains analytical, not measured timestamps.
The current offline adapter rebuilds ready-set policy instances for each selection,
so host scheduling time grows roughly quadratically with job count; start with
1,000 rather than millions of events.

For a longer two-stage prefill/decode trace on the H100-inspired pair:

```bash
make benchmark EXPERIMENT=h100 BENCHMARK_ARGS="--workload prefill-decode -- --decode-tokens 100"
```

That workload is still one sequential request, so increasing its length does not
necessarily create different policy results.

Extra workload parameters are identical across all policies. The batch-managed
options `--policy`, `--output-dir`, `--backend`, `--config-dir`, `--workload`,
`--timeout`, and listing flags cannot be forwarded after `--`; set supported batch
overrides before the separator instead. All runs use the same saved hardware config
snapshot under `_config/`, so source config edits during a batch do not change its
inputs.

The runner uses the existing `compare_results.py` comparison script to produce
the chart, including an analytical latency panel for the Jetson experiment.
These are synthetic, modeled results, not hardware-calibrated measurements or
predictions of measured device performance.

To invoke the Python batch runner directly:

```bash
docker compose run --rm scheduler python -m experiments.compare_schedulers --experiment jetson
```

Compose bind-mounts the source, so Python-only source changes require no image
rebuild.

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
