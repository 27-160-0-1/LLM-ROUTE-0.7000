# Honest held-out evaluation: train the whole chain on Train (1,760) only,
# score Dev (880) with the public lookup stripped.  Deployment artifacts are
# untouched — everything is built under reports/holdout/.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "$root\src"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\012\SKT LLM\.venv\Scripts\python.exe"
$out = "$root\reports\holdout"
New-Item -ItemType Directory -Force $out | Out-Null
$art = "$out\learned-router.v1.json"

Write-Host "[holdout] 1) linear head (Train only)"
& $py tools/train_learned_router_gpu.py `
  --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json `
  --validation-input data/materialized/dev/inputs.json --validation-outcomes data/dev/outcomes.json `
  --artifact $art --report "$out\gpu-report.json" `
  --word-bins 8192 --char-bins 8192 --alphas 30 --blend-weights 0.75 --context-limits 1000000
if ($LASTEXITCODE -ne 0) { throw "linear stage failed" }

Write-Host "[holdout] 2) augmentation (Train only)"
& $py tools/build_router_augmentation.py --artifact $art `
  --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "augmentation stage failed" }

Write-Host "[holdout] 3) meta GBM (Train only)"
& $py tools/build_meta_gbm.py --artifact $art `
  --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "meta stage failed" }

Write-Host "[holdout] 4) held-out Dev score (no lookup)"
& $py tools/holdout_eval.py --artifact $art --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json
