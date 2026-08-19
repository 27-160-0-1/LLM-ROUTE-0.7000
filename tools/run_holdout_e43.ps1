# Honest held-out evaluation of the E43 sweep candidate (cand0): train the whole chain on
# Train (1,760) only with the candidate constants, score Dev (880) with the public lookup
# stripped.  Deployment artifacts are untouched — everything is built under reports/holdout_e43/.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "$root\src"
$env:PYTHONUTF8 = "1"
# E43 cand0 constants (env hooks read by the build tools)
$env:ROUTER_FAM_W = "0.15"
$env:ROUTER_CONF_SCALE = "0.25"
$env:ROUTER_BLEND_FAST = "0.6"
$env:ROUTER_BLEND_BALANCED = "0.45"
$env:ROUTER_BLEND_PREMIUM = "0.3"
$env:ROUTER_RIDGE_ALPHA = "10"
$env:ROUTER_GAIN_ALPHA = "0.5"
$env:ROUTER_RANK_BETA = "0.4"
$py = "C:\Users\012\SKT LLM\.venv\Scripts\python.exe"
$out = "$root\reports\holdout_e43"
New-Item -ItemType Directory -Force $out | Out-Null
$art = "$out\learned-router.v1.json"

Write-Host "[holdout-e43] 1) linear head (Train only, alpha 10, legacy blend 0.9)"
& $py tools/train_learned_router_gpu.py `
  --input data/materialized/train/inputs.json --outcomes data/train/outcomes.json `
  --validation-input data/materialized/dev/inputs.json --validation-outcomes data/dev/outcomes.json `
  --artifact $art --report "$out\gpu-report.json" `
  --word-bins 8192 --char-bins 8192 --alphas 10 --blend-weights 0.9 --context-limits 1000000
if ($LASTEXITCODE -ne 0) { throw "linear stage failed" }

Write-Host "[holdout-e43] 2) augmentation (Train only)"
& $py tools/build_router_augmentation.py --artifact $art `
  --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "augmentation stage failed" }

Write-Host "[holdout-e43] 3) meta GBM (Train only)"
& $py tools/build_meta_gbm.py --artifact $art `
  --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "meta stage failed" }

Write-Host "[holdout-e43] 4) held-out Dev score (no lookup)"
& $py tools/holdout_eval.py --artifact $art --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json
