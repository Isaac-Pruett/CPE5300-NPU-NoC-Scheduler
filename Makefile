export LOCAL_UID := $(shell id -u)
export LOCAL_GID := $(shell id -g)
RUN = docker compose run --rm scheduler
COMPARE_RUNS ?= results/raw/fifo-demo results/raw/sjf-demo
COMPARE_OUTPUT ?= results/policy_comparison.png
COMPARE_ARGS ?=
POLICY_DEMO_DIR ?= results/raw/policy-demo

.PHONY: build simulate speculative test profile inspect sweep plot compare policy-demo shell
build:
	docker compose build

simulate:
	$(RUN) python -m astra.run --workload prefill-decode

speculative:
	$(RUN) python -m astra.run --workload speculative --backend congestion-unaware

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
	$(RUN) python -m astra.run --workload policy-demo --policy fifo --output-dir "$(POLICY_DEMO_DIR)/fifo"
	$(RUN) python -m astra.run --workload policy-demo --policy sjf --output-dir "$(POLICY_DEMO_DIR)/sjf"
	$(RUN) python compare_results.py "$(POLICY_DEMO_DIR)/fifo" "$(POLICY_DEMO_DIR)/sjf" --output "$(POLICY_DEMO_DIR)/comparison.png"

shell:
	$(RUN) bash

clean:
	rm -rf scheduler/results/raw
