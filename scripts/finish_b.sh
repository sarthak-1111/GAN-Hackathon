#!/usr/bin/env bash
# Wait for PDF realism run 2, select the PDF run (validation score), then train predictor (b) seeds 1 and 2.
cd /d/GAN
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=/d/GAN
until [ -f artifacts/models/realism_pdf_r1g10/generator_ema.sha256 ]; do sleep 30; done
sleep 20
date
.venv/Scripts/python.exe -m forge.finalize select 2>&1 | grep -v "Warn\|No conda"
ls artifacts/models/
.venv/Scripts/python.exe -u -m forge.train_predictor configs/predictor_b_s1.yaml > logs/predictor_b_s1.log 2>&1 &
p1=$!
.venv/Scripts/python.exe -u -m forge.train_predictor configs/predictor_b_s2.yaml > logs/predictor_b_s2.log 2>&1 &
p2=$!
wait $p1; e1=$?
wait $p2; e2=$?
date
echo "predictor b exit codes: $e1 $e2"
grep "\] step" logs/predictor_b_s1.log | tail -2
grep "\] step" logs/predictor_b_s2.log | tail -2
