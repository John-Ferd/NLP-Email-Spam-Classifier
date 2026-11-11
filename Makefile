PY ?= python
DATA_URL = https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip
RUN = cd scripts && $(PY)

.PHONY: setup data test train calibrate conformal drift all smoke lint

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	mkdir -p data
	curl -L -o data/enron_spam_data.zip $(DATA_URL)
	cd data && $(PY) -c "import zipfile; zipfile.ZipFile('enron_spam_data.zip').extractall('.')"

test:
	$(PY) -m pytest -q

train:
	$(RUN) train.py

calibrate:
	$(RUN) run_calibration.py

conformal:
	$(RUN) run_conformal.py

drift:
	$(RUN) run_drift.py

all: train calibrate conformal drift

smoke:   # synthetic data; writes to .smoke-artifacts/ so real results are untouched
	cd scripts && SPAMGUARD_ARTIFACTS=../.smoke-artifacts $(PY) train.py --smoke
	cd scripts && SPAMGUARD_ARTIFACTS=../.smoke-artifacts $(PY) run_conformal.py --smoke

lint:
	$(PY) -m ruff check src scripts tests
