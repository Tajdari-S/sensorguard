PYTHON ?= python3
GPU ?= 0

.PHONY: validate roofline-test roofline-smoke

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
