#!/usr/bin/env bash
# Phase 5 after all models are trained: decision rule, cards, tidy-up, test-split inference, tests, final validation.
# (GAN diagnostics were computed per technique as soon as each realism model was final:
#  python -m forge.finalize diagnostics xrd exafs ; then pdf on GPU 2; results are cached with model hashes.)
set -e
cd /d/GAN
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=/d/GAN
.venv/Scripts/python.exe -m forge.finalize downstream      # sets A/B/C, decision rule, ships the predictor
.venv/Scripts/python.exe -m forge.finalize diagnostics     # reuses cached per-technique results if models unchanged
.venv/Scripts/python.exe -m forge.finalize cards           # model cards and dataset card
.venv/Scripts/python.exe -m forge.finalize tidy            # intermediate checkpoints -> artifacts/models/work/
.venv/Scripts/python.exe -m forge.test_inference           # predict.py / realism_infer.py on the test split
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
.venv/Scripts/python.exe artifacts/scripts/validate_dataset.py
