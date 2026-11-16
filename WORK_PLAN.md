# Your work plan: code and tests to build next

A concrete, ordered checklist for moving this project from "synthetic baseline"
to "real phase-aware NoC experiment." Do the tasks top to bottom — each one
depends on the one before it. Every task tells you the file to edit, exactly
what to do, the test to write, and the "done when" check.

Read `README_SUPPORT.md` and `NEXT_STEPS.md` first if you haven't. This file is
the hands-on version of `NEXT_STEPS.md` steps 3–4, plus a hardware-profile system.

---

## How to work each task

1. Make a branch: `git checkout -b task-N-short-name`.
2. Write the test first (it should fail — that's "red").
3. Implement until the test passes ("green") and `make test` is clean.
4. Commit with a clear message. Move to the next task.

Run tests with:

```bash
docker compose run --rm scheduler python -m unittest discover -s tests -v
# or
make test
```

---

## Task 0 — Confirm the baseline runs (do first, ~20 min)

Prove the current pipeline works so you have a known-good starting point.

- **Run:** `docker compose build`, then `make test simulate speculative`.
- **Check:** each `scheduler/results/raw/<timestamp>-.../summary.json` has
  `"status": "completed"` and a `ranks` entry for both `0` and `1`.
- **Also do:** open one real `simulator.log` and skim it. You'll need to know
  what timing lines ASTRA-sim actually prints for Task 4.

**Done when:** tests pass and both simulations complete without code changes.
**Files:** none (read-only).

---

## Task 1 — Add request identity to the event model

**Why:** you can't tell requests apart yet. Everything else needs this.

- **Edit `scheduler/speculative/event_model.py`:**
  - Add to the `Event` dataclass: `request_id: int | None = None` and
    `token_index: int | None = None`.
  - Add both to `to_dict()` (only emit when not `None`, like the other optional
    fields) and read them in `from_dict()`.
  - In `Event.validate()`: if present, require `request_id >= 0` and
    `token_index >= 0`.
- **Edit `scheduler/workloads/prefill_decode.py`:**
  - In `build_prefill_decode`, set `request_id=0` on every event and
    `token_index=step`.
- **Edit `scheduler/tracegen/spec_to_chakra.py`:**
  - In `_convert_event`, emit `request_id` and `token_index` as Chakra
    attributes when present. Copy the exact pattern used for `phase`
    (`AttributeProto(name="request_id", uint32_val=event.request_id)`).

**Test (new method in `scheduler/tests/test_workflows.py`):**
- Build a prefill/decode trace; assert every event has `request_id == 0`.
- Round-trip through `write_json` / `read_json`; assert fields survive.
- Convert to Chakra, decode the `.et`, assert the attributes are present with
  correct values. Copy the decode loop from
  `test_json_and_chakra_preserve_dependencies_phases_and_payloads`.

**Done when:** the fields round-trip through JSON *and* appear in the decoded
Chakra trace, and all existing tests still pass.
**Difficulty:** easy.

---

## Task 2 — Generate multiple, overlapping requests

**Why:** with one request there is never contention, so there is nothing to
schedule. (NEXT_STEPS step 3.)

- **Edit `scheduler/workloads/prefill_decode.py`:**
  - Add a `num_requests: int = 1` parameter (validate `>= 1`).
  - Wrap the existing per-step loop so it runs once per request, each with a
    distinct `request_id`.
  - **GOTCHA — unique tags.** `EventTrace.validate()` matches sends to receives
    by their `tag`. If two requests reuse a tag, validation fails with
    "unmatched communication." Make every tag unique per request:
    `f"req{r}_{label}_activation"` and `f"req{r}_{label}_feedback"`.
  - Keep each request's own autoregressive chain (its step N waits on its step
    N-1). Do **not** make request A depend on request B.
- **Edit `add_workload_arguments`** (same file) and
  **`scheduler/astra/run.py`** to accept `--num-requests` and pass it through.

**Tests:**
- `num_requests=3` produces exactly 3 distinct `request_id` values.
- All communication tags are unique: collect them, assert
  `len(set(tags)) == len(tags)`.
- `trace.validate()` passes.
- Per request, each decode step depends on that request's own previous RECV
  (adapt `test_autoregressive_steps_wait_for_feedback` to loop over request IDs).

**Done when:** a 3-request trace validates, has unique tags, and preserves
per-request autoregressive ordering.
**Difficulty:** medium.

---

## Task 3 — End-to-end simulator smoke test

**Why:** no test currently runs ASTRA-sim. This catches breakage between
"trace encodes" and "simulator accepts it."

- **Refactor `scheduler/astra/run.py`:** extract a callable function, e.g.
  `run_simulation(trace, backend, output_dir, timeout) -> dict` that does the
  work currently inside `main()` after arg parsing. `main()` then just parses
  args and calls it. This makes the runner testable.
- **New file `scheduler/tests/test_simulation.py`:**
  - Skip cleanly when not in Docker:
    ```python
    import shutil, unittest
    @unittest.skipIf(
        shutil.which("AstraSim_Analytical_Congestion_Aware") is None,
        "ASTRA-sim not available; run inside Docker",
    )
    ```
  - Run a tiny workload (`decode_tokens=1`) into a temp dir; assert the returned
    summary / `summary.json` has `status == "completed"` and both ranks present.

**Done when:** `make test` runs this and it either completes the sim (in Docker)
or skips cleanly (on host).
**Difficulty:** medium. Independent of Tasks 1–2, so you can do it early.

---

## Task 4 — Extract per-request / per-token metrics

**Why:** the runner only parses total cycles. You need TTFT (time to first
token) and inter-token latency to compare policies. (NEXT_STEPS step 3.)

- **First, investigate (don't assume):** read a real `simulator.log` from Task 0.
  Write down exactly what per-event/per-rank timing the analytical backend
  prints. The current `completed_ranks()` regex only captures `sys[N] finished`
  totals and `GPU time`.
- **Edit `scheduler/astra/run.py`:** add a metrics function that computes TTFT
  and inter-token latency from whatever timeline is available, and write them
  into `summary.json`. Define units explicitly (cycles vs. ns vs. us).
- If the analytical backend does not expose enough per-event timing, **say so in
  the summary** and compute only what you honestly can. Do not fabricate numbers.

**Test:**
- Feed a small, **known** synthetic log string to the metrics function and
  assert the computed TTFT / inter-token values. Mirror
  `test_incomplete_simulation_is_not_success`, which already parses hardcoded
  log strings.

**Done when:** metrics are computed from a known timeline and verified by a unit
test, and real runs populate them in `summary.json`.
**Difficulty:** medium–hard.

---

## Task 5 — Baseline vs. phase-aware comparison (the real experiment)

**Why:** this is the project's actual contribution. (NEXT_STEPS step 4.)
Do it only after 1–4.

- **Decide the scheduling point first:** request admission, endpoint
  transmission, or router arbitration. These are different experiments. Read the
  ASTRA-sim system-config and analytical-network docs (links in
  `README_SUPPORT.md` section 10) to see what the backend lets you control.
- **Run the same multi-request workload under two configs:** a phase-unaware
  baseline (FIFO or round-robin) vs. a phase-aware policy (e.g. decode priority
  with aging so prefill never starves). Compare the Task 4 metrics.
- **Keep simulator-side changes in version control** and apply them at Docker
  build time. Do not rely on edits made inside a disposable container.

**Tests:**
- Controlled contention: two overlapping requests; assert service order differs
  between baseline and phase-aware for the intended reason.
- Starvation guard: sustained decode load; assert every prefill request still
  completes.

**Done when:** baseline and phase-aware runs differ measurably on the same
workload, and prefill never starves.
**Difficulty:** hard.

---

## Order and dependency summary

| Task | File(s) | Difficulty | Needs |
| --- | --- | --- | --- |
| 0 Baseline run | (none) | trivial | — |
| 1 Request identity | event_model, prefill_decode, spec_to_chakra | easy | 0 |
| 2 Concurrent requests | prefill_decode, run | medium | 1 |
| 3 Sim smoke test | new test_simulation.py, run | medium | 0 |
| 4 Metrics | run | medium–hard | 2, 3 |
| 5 Policy comparison | configs, run, docs | hard | 4 |
| H Hardware profiles | run, configs/ (see below) | medium | 0 |

Start 0 → 1. Task 3 and Task H are independent of 1–2, so do them early if you
want quick wins.

---

# Task H — Plug-and-play hardware profiles (HGX / TPU / RSC)

**Goal:** be able to say `--hardware hgx-h100` or `--hardware tpu-v5` or
`--hardware meta-rsc` and have the simulator load that machine's topology,
bandwidth, latency, and compute numbers — without editing code each time.

## Why this is a config problem, not a code problem

ASTRA-sim is already hardware-parameterized. A "machine" here is just the trio of
config files the runner feeds it:

- `network.yml` — topology, number of endpoints, bandwidth, latency
- `system.json` — scheduling policy, endpoint delay, collective implementations,
  local memory bandwidth, peak perf
- `remote_memory.json` — memory-expansion model

Today `scheduler/astra/run.py` has **one hardcoded path**:

```python
CONFIG = Path(__file__).resolve().parent / "configs" / "two_rank"
```

"Plug and play" = turn that single hardcoded directory into a **selectable set of
named profile directories**, each holding one machine's three files.

> **Honesty note:** the values you put in these profiles are *modeling
> approximations* of public specs, not the real proprietary interconnects. Label
> every number with its source and call these "HGX-like" / "TPU-like" profiles.
> Do not claim they reproduce the real hardware.

## Step H1 — Create a profiles directory

```
scheduler/astra/configs/
  two_rank/            # keep the existing default
  profiles/
    hgx-h100/
      network.yml
      system.json
      remote_memory.json
    tpu-v5e/
      network.yml
      system.json
      remote_memory.json
    meta-rsc/
      network.yml
      system.json
      remote_memory.json
```

Copy the three `two_rank` files into each new folder as a starting point, then
change the numbers to match each machine's public specs.

**What differs between the profiles (the parts you research and set):**

| Field | File | What it represents |
| --- | --- | --- |
| `topology` | network.yml | e.g. FullyConnected (NVLink-like island), Ring, Switch, Mesh2D |
| `npus_count` | network.yml | number of accelerators / tiles in the domain |
| `bandwidth` | network.yml | per-link GB/s (NVLink vs ICI vs Ethernet/IB differ a lot) |
| `latency` | network.yml | per-link latency in ns |
| `local-mem-bw` | system.json | on-device memory bandwidth |
| `peak-perf` | system.json | modeled accelerator compute |
| `memory-type` | remote_memory.json | NO_MEMORY_EXPANSION or an expansion model |

**Rough modeling direction (verify against current public specs before using):**

- **HGX (NVIDIA, NVLink/NVSwitch domain):** high per-link bandwidth, small
  low-latency fully-connected or switched island of 8 GPUs. Model as
  `FullyConnected` or `Switch`, `npus_count: [8]`, high `bandwidth`.
- **Cloud TPU (Google, ICI):** TPUs wired in a torus/mesh with inter-chip
  interconnect. Model as a mesh/ring, more endpoints, different bandwidth/latency.
- **Meta RSC (cluster scale):** many nodes over a data-center fabric
  (Ethernet/InfiniBand), so lower per-link bandwidth and higher latency than an
  on-package NVLink domain, with many more endpoints — a multi-dimensional
  topology.

Put the exact numbers and your source URL in a comment at the top of each
`network.yml`.

## Step H2 — Make the runner select a profile

**Edit `scheduler/astra/run.py`:**

1. Replace the single hardcoded `CONFIG` with a lookup:
   ```python
   CONFIGS_ROOT = Path(__file__).resolve().parent / "configs"

   def resolve_hardware(name: str) -> Path:
       if name == "two_rank":
           path = CONFIGS_ROOT / "two_rank"
       else:
           path = CONFIGS_ROOT / "profiles" / name
       required = ("network.yml", "system.json", "remote_memory.json")
       missing = [f for f in required if not (path / f).exists()]
       if missing:
           raise FileNotFoundError(
               f"hardware profile '{name}' missing files: {missing} (looked in {path})"
           )
       return path
   ```
2. Add an argument:
   ```python
   parser.add_argument("--hardware", default="two_rank",
                       help="Hardware profile name (folder under configs/ or configs/profiles/)")
   ```
3. Use `config_dir = resolve_hardware(args.hardware)` where the old `CONFIG`
   was used (the `shutil.copyfile(CONFIG / name, ...)` lines).

Now `--hardware hgx-h100` just works.

> **Watch out:** workloads currently assume **exactly two ranks**
> (`prefill_decode.py` and `run.py`). A profile with `npus_count: [8]` needs the
> workload generator to produce events for all 8 ranks, or the run will fail. So
> either (a) keep profiles at 2 endpoints until Task 2's multi-rank work is done,
> or (b) tie `num_requests` / rank count to the profile. Decide this explicitly.

## Step H3 — Tests for the hardware system

**New file `scheduler/tests/test_hardware_profiles.py`:**

- **Discovery test:** loop over every folder in `configs/profiles/`, assert each
  has all three required files.
- **Resolver test:** `resolve_hardware("hgx-h100")` returns a path containing the
  three files; `resolve_hardware("does-not-exist")` raises `FileNotFoundError`.
- **Parse test:** load each profile's `network.yml` (YAML) and the two JSON files
  and assert they parse and contain the required keys (`topology`, `npus_count`,
  `bandwidth`, `latency` / `scheduling-policy` / `memory-type`). This catches a
  typo in a profile before a 2-minute simulator run does.
- **Consistency test:** assert `len(network["bandwidth"]) == len(network["topology"])`
  and that `npus_count` matches the ranks the workload will generate.

**Pattern** — parametrize over discovered profiles so new profiles are tested
automatically:

```python
import json, unittest
from pathlib import Path
import yaml  # add pyyaml to requirements if not present

PROFILES = Path(__file__).resolve().parents[1] / "astra" / "configs" / "profiles"

class HardwareProfileTests(unittest.TestCase):
    def test_every_profile_is_complete_and_parses(self):
        profiles = [p for p in PROFILES.iterdir() if p.is_dir()]
        self.assertTrue(profiles, "no hardware profiles found")
        for profile in profiles:
            with self.subTest(profile=profile.name):
                net = yaml.safe_load((profile / "network.yml").read_text())
                sysc = json.loads((profile / "system.json").read_text())
                mem = json.loads((profile / "remote_memory.json").read_text())
                for key in ("topology", "npus_count", "bandwidth", "latency"):
                    self.assertIn(key, net)
                self.assertIn("scheduling-policy", sysc)
                self.assertIn("memory-type", mem)
                self.assertEqual(len(net["bandwidth"]), len(net["topology"]))
```

> **Note:** check whether `pyyaml` is already available in the image before using
> `yaml` in a test (ASTRA-sim reads the YAML itself; the Python side may not have
> pyyaml). If not, either add `pyyaml` to `scheduler/requirements.txt` or parse
> `network.yml` as text. JSON files need no extra dependency.

## Step H4 — Make it convenient (optional)

Add Makefile shortcuts so running a machine is one word:

```makefile
sim-hgx:
	$(RUN) python -m astra.run --workload prefill-decode --hardware hgx-h100

sim-tpu:
	$(RUN) python -m astra.run --workload prefill-decode --hardware tpu-v5e

sim-rsc:
	$(RUN) python -m astra.run --workload prefill-decode --hardware meta-rsc
```

And record which hardware profile was used in `summary.json` (add
`summary["hardware"] = args.hardware` in `run.py`) so every result is traceable
to its machine.

## Hardware task "done when"

- `--hardware <name>` loads the right config trio with a clear error for a bad
  name.
- Every profile is auto-discovered and validated by a test before any run.
- `summary.json` records which hardware profile produced each result.
- Each profile's numbers carry a source comment and are labeled "like" the real
  machine, not claimed to reproduce it.

---

## Final reminders

- Follow `NEXT_STEPS.md` literally; each step's "Done when:" is your definition
  of finished.
- Do the experiment spec (NEXT_STEPS step 2: "a rank = a tile," pick a topology,
  pick link numbers) before expanding the model — Task H profiles are where that
  decision gets written down as real config.
- Never relabel the `experiments/sweep_speculation.py` formula output as
  simulation results.
- Keep synthetic and measured/approximated numbers clearly distinct in every
  summary and report.
