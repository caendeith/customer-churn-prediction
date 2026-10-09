# Optional convenience wrapper. The primary documented path is plain
# `python run_all.py` -- make is not required and may not be installed.

PYTHON ?= python

.PHONY: all run sample download predict clean test

all: run

run:
	$(PYTHON) run_all.py

sample:
	$(PYTHON) run_all.py --sample

download:
	$(PYTHON) run_all.py --download

predict:
	$(PYTHON) predict.py --input data/sample/predict_example.csv

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; shutil.rmtree('reports/figures', ignore_errors=True); [os.remove(f) for f in ['models/churn_model.joblib','reports/model_report.md','reports/metrics.json'] if os.path.exists(f)]"
