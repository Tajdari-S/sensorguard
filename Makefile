PYTHON ?= python3
GPU ?= 0
RUN_ID ?= smoke
WORKLOAD_CMD ?= $(PYTHON) scripts/loggers/calib_workload.py --mode gemm --device cuda:$(GPU) --duration-s 60

.PHONY: validate test data-audit baseline-audit figures roofline-test roofline-smoke inventory supervised-run calibration-report

validate:
	$(PYTHON) scripts/check_workload_coverage.py
	$(PYTHON) scripts/validate_preregistration.py configs/preregistration.yaml

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

data-audit:
	$(PYTHON) scripts/analysis/repair_label_metadata.py \
		results/e2_labels_night1.csv results/e2_labels_combined.csv
	$(PYTHON) -c "import pandas as pd; d=pd.read_csv('results/e2_labels_combined.csv'); assert d.gpu_uuid.notna().all(); assert d.run_id.is_unique"

baseline-audit:
	$(PYTHON) scripts/analysis/train_baseline.py --labels results/e2_labels_combined.csv --group-by run_id --output results/evaluation/baseline-run-grouped.json
	$(PYTHON) scripts/analysis/train_baseline.py --labels results/e2_labels_combined.csv --group-by gpu_uuid --output results/evaluation/baseline-gpu-grouped.json
	$(PYTHON) scripts/analysis/train_baseline.py --labels results/e2_labels_combined.csv --group-by family --output results/evaluation/baseline-family-grouped.json
	$(PYTHON) scripts/analysis/train_baseline.py --labels results/e2_labels_combined.csv --group-by collection_day --output results/evaluation/baseline-day-grouped.json

figures:
	MPLCONFIGDIR=/tmp/sensorguard-mpl $(PYTHON) scripts/figures/make_existing_figures.py

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
