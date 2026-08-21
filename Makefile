.PHONY: all lint test quality-report calibrate crossval equifinality \
        residuals hybrid explain humidity counterfactual reproduce \
        dashboard clean data

# `all` is the fast, CI-safe path -- the same two steps ci.yml runs on
# every push. It is deliberately NOT the full research pipeline: lint +
# test finish in under a minute and a green run is a precondition for
# trusting anything `reproduce` below would produce. `reproduce` is the
# separate, slow target L9.1 actually verifies against -- see its own
# comment for why the two are not merged into one `all`.
all: lint test

lint:
	ruff check src tests dashboard
	mypy src dashboard

test:
	pytest --cov=cooling_twin --cov-report=term-missing --cov-fail-under=80

# BDG2 is ~1.6 GB and distributed over git-lfs, not something a Makefile
# target should silently fetch on every clean-clone run -- see
# 04_DATA_CONTRACT.md for the pull command and ~/data/bdg2 layout this
# project expects (never /mnt/c, per CLAUDE.md SS10). This target is a
# pointer to that document, not a downloader: `scripts/download_data.py`
# is still an empty stub (unchanged from M2 -- BDG2 was pulled by hand
# once, per 07_PROGRESS.md's L2.1 entry, and every later stage reads it
# from the fixed local path load.py expects), and a target that shells
# out to an empty script would exit 0 while doing nothing.
data:
	@echo "BDG2 is not auto-downloaded. Follow 04_DATA_CONTRACT.md's git-lfs"
	@echo "pull instructions once, into ~/data/bdg2, then re-run this target's"
	@echo "dependents. This target only checks the data is where load.py"
	@echo "expects it, per config/buildings.yaml's data_root."
	@test -d "$${BDG2_ROOT:-$$HOME/data/bdg2}" || \
		(echo "missing: $${BDG2_ROOT:-$$HOME/data/bdg2} -- see 04_DATA_CONTRACT.md" && exit 1)

quality-report:
	python scripts/generate_quality_report.py

# M6. `calibrate` and `crossval`/`equifinality` are separate targets
# (not chained) because they answer different questions on the SAME
# frozen artifact: crossval scores generalisation inside the training
# year, equifinality asks whether a different parameter set fits just
# as well. Chaining them would suggest one is a prerequisite step of
# the other rather than two independent probes of one calibration run.
calibrate:
	python scripts/run_calibration.py --config config/calibration.yaml

crossval:
	python scripts/run_crossval.py --config config/calibration.yaml

equifinality:
	python scripts/run_equifinality.py --config config/calibration.yaml

# M7. Each script reads FROZEN 2016 parameters from calibrate's
# artifact and never re-fits them (see each script's own module
# docstring) -- residuals/hybrid/explain/humidity can run in any order
# relative to each other, only after calibrate.
residuals:
	python scripts/analyse_residuals.py

hybrid:
	python scripts/fit_hybrid_residual.py

explain:
	python scripts/compare_explanations.py

humidity:
	python scripts/compare_site_humidity.py

# M8. Order matters only for the dashboard, which reads what these
# write; each script is independently runnable and each writes its own
# JSON artifact. All three refuse to touch 2017 (ADR-002).
counterfactual:
	python scripts/compare_correlation_intervention.py
	python scripts/run_counterfactual.py
	python scripts/validate_intervals.py

# L9.1's full chain, deliberately NOT part of `all`. Two reasons:
#   1. `open_test_set.py` (ADR-002's one-time 2017 opening) and its
#      re-read scripts (`reread_hybrid_2017.py`, the `--reread-test-set`
#      flag on `investigate_ushape.py`) are excluded on purpose -- a
#      Makefile target that could re-run them would turn "reproduce the
#      project" into "spend the test set again", which ADR-002 forbids
#      by construction, not by discipline. Their numbers are already on
#      disk in reports/calibration_runs/ and are not regenerated here.
#   2. `calibrate` alone runs a differential-evolution search per
#      building; the full chain below is measured in HOURS, not
#      minutes (CLAUDE.md SS10: warn before pegging CPU past ~1 hour).
#      It is a target you run once overnight to check the artifacts on
#      disk still reproduce, not a step in ordinary iteration.
reproduce: data calibrate crossval equifinality quality-report residuals hybrid explain humidity counterfactual
	@echo "reproduce: pipeline complete. Compare reports/calibration_runs/*.json"
	@echo "against the numbers quoted in reports/0*.md -- any drift means an"
	@echo "artifact was hand-edited or a script changed behaviour silently."

# Interactive (Streamlit) dashboard -- replaces the old static-HTML
# `build_dashboard.py` build. Reads run artifacts and recomputes hourly
# series live against BDG2; see dashboard/app.py and dashboard/data.py.
dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
