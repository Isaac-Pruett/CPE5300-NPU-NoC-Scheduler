export LOCAL_UID := $(shell id -u)
export LOCAL_GID := $(shell id -g)
RUN = docker compose run --rm scheduler
COMPARE_RUNS ?= results/raw/fifo-demo results/raw/sjf-demo
COMPARE_OUTPUT ?= results/policy_comparison.png
COMPARE_ARGS ?=
POLICY_DEMO_DIR ?= results/raw/policy-demo
JETSON_DEMO_DIR ?= results/raw/jetson-demo
CONFIG_DIR ?= two_rank
EXPERIMENT ?= jetson
BENCHMARK_DIR ?=
BENCHMARK_ARGS ?=

.PHONY: build simulate speculative test profile inspect sweep plot compare policy-demo jetson-demo benchmark configs policies shell
build:
	docker compose build

simulate:
	$(RUN) python -m astra.run --workload prefill-decode --config-dir "$(CONFIG_DIR)"

speculative:
	$(RUN) python -m astra.run --workload speculative --backend congestion-unaware --config-dir "$(CONFIG_DIR)"

configs:
	$(RUN) python -m astra.run --list-configs

policies:
	$(RUN) python -m astra.run --list-policies

test:
	$(RUN) python -m unittest discover -s tests -v

profile:
	$(RUN) python profile_workload.py

inspect:
	$(RUN) python inspect_trace.py traces/kineto_trace_rank0.json

sweep:
	$(RUN) python experiments/sweep_speculation.py

plot:
	$(RUN) python plot_results.py

compare:
	$(RUN) python compare_results.py $(COMPARE_RUNS) --output "$(COMPARE_OUTPUT)" $(COMPARE_ARGS)

policy-demo:
	$(RUN) python -m astra.run --workload policy-demo --policy fifo --config-dir "$(CONFIG_DIR)" --output-dir "$(POLICY_DEMO_DIR)/fifo"
	$(RUN) python -m astra.run --workload policy-demo --policy sjf --config-dir "$(CONFIG_DIR)" --output-dir "$(POLICY_DEMO_DIR)/sjf"
	$(RUN) python compare_results.py "$(POLICY_DEMO_DIR)/fifo" "$(POLICY_DEMO_DIR)/sjf" --output "$(POLICY_DEMO_DIR)/comparison.png"

jetson-demo:
	$(RUN) python -m astra.run --workload single-processor-demo --policy fifo --config-dir jetson_orin_nano --output-dir "$(JETSON_DEMO_DIR)/fifo"
	$(RUN) python -m astra.run --workload single-processor-demo --policy sjf --config-dir jetson_orin_nano --output-dir "$(JETSON_DEMO_DIR)/sjf"
	$(RUN) python compare_results.py "$(JETSON_DEMO_DIR)/fifo" "$(JETSON_DEMO_DIR)/sjf" --output "$(JETSON_DEMO_DIR)/comparison.png"

benchmark:
	$(RUN) python -m experiments.compare_schedulers --experiment "$(EXPERIMENT)" $(if $(BENCHMARK_DIR),--output-dir "$(BENCHMARK_DIR)") $(BENCHMARK_ARGS)

shell:
	$(RUN) bash

clean:
	rm -rf scheduler/results/raw
