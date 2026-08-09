.PHONY: all lint test quality-report clean

# `data`, `calibrate`, and `report` targets are not listed here yet --
# scripts/download_data.py and scripts/run_calibration.py are still
# empty stubs (M5/M6 build them out). A target that shells out to an
# empty script would exit 0 and look like it worked while doing
# nothing -- worse than no target at all.
all: lint test

lint:
	ruff check src tests
	mypy src

test:
	pytest --cov=cooling_twin --cov-report=term-missing --cov-fail-under=80

quality-report:
	python scripts/generate_quality_report.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
