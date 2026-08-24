PYTHON ?= python3
GPU ?= 0
RUN_ID ?= smoke
WORKLOAD_CMD ?= $(PYTHON) scripts/loggers/calib_workload.py --mode gemm --device cuda:$(GPU) --duration-s 60

.PHONY: validate roofline-test roofline-smoke inventory supervised-run calibration-report

validate:
	$(PYTHON) scripts/check_workload_coverage.py
	$(PYTHON) scripts/validate_preregistration.py configs/preregistration.yaml

roofline-test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

roofline-smoke:
	$(PYTHON) scripts/roofline/benchmark_kernels.py \
		--case gemm_1024 --case copy --device cuda:$(GPU) \
		--warmup 3 --iterations 5 \
		--output results/roofline/smoke-gpu$(GPU).json

inventory:
	bash scripts/capture_inventory.sh results/inventory-$$(hostname)

supervised-run:
	$(PYTHON) scripts/loggers/supervisor.py --run-id $(RUN_ID) \
		--workload-cmd "$(WORKLOAD_CMD)" --gpu-index $(GPU) --sensors nvml,dcgm

calibration-report:
	$(PYTHON) scripts/loggers/calibration_report.py --runs-dir data/runs \
		--output results/e1_calibration_report.md
