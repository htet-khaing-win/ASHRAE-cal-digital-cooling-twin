#!/usr/bin/env bash
# scripts/bootstrap.sh — create the full project skeleton
set -euo pipefail

mkdir -p cooling-twin && cd cooling-twin

# package
mkdir -p src/cooling_twin/{data,models,calibration,analysis,twin,viz}
for pkg in "" data models calibration analysis twin viz; do
  touch "src/cooling_twin/${pkg}/__init__.py"
done

# module stubs
touch src/cooling_twin/data/{load,quality,weather,schema,select}.py
touch src/cooling_twin/models/{rc,chiller,pump,tower,plant}.py
touch src/cooling_twin/calibration/{metrics,baseline,sensitivity,optimize}.py
touch src/cooling_twin/analysis/{residual,hybrid}.py
touch src/cooling_twin/twin/{counterfactual,uncertainty}.py
touch src/cooling_twin/viz/plots.py

# tests
mkdir -p tests/fixtures
touch tests/{__init__,test_seed,test_quality,test_rc,test_metrics}.py

# config, scripts, reports, notebooks, data
mkdir -p config scripts notebooks reports/figures
mkdir -p data/{raw,interim,processed}
touch config/{buildings,cleaning,calibration}.yaml
touch scripts/{check_env,bootstrap,download_data,run_calibration}.py
touch reports/{01_data_quality,02_calibration,03_residual_analysis}.md
touch README.md Makefile pyproject.toml environment.yml .gitignore
mkdir -p .github/workflows && touch .github/workflows/ci.yml

git init -q
echo "Scaffold created. Files: $(find . -type f -not -path './.git/*' | wc -l)"