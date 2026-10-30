export LOCAL_UID := $(shell id -u)
export LOCAL_GID := $(shell id -g)
RUN = docker compose run --rm scheduler

.PHONY: build simulate speculative test profile inspect sweep plot shell
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

shell:
	$(RUN) bash
