# Rebuild the deployment artifacts with the E43 constants (Train+Dev 2,640), regenerate the public
# lookup and split the heavy block.  Previous artifacts are kept as *.e27.bak.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "$root\src"
$env:PYTHONUTF8 = "1"
# E43 cand0 constants + re-calibrated safety ratios (env hooks read by the build tools)
$env:ROUTER_FAM_W = "0.15"
$env:ROUTER_CONF_SCALE = "0.25"
$env:ROUTER_BLEND_FAST = "0.6"
$env:ROUTER_BLEND_BALANCED = "0.45"
$env:ROUTER_BLEND_PREMIUM = "0.3"
$env:ROUTER_RIDGE_ALPHA = "10"
$env:ROUTER_GAIN_ALPHA = "0.5"
$env:ROUTER_RANK_BETA = "0.4"
$env:ROUTER_SAFETY_FAST = "0.98"
$env:ROUTER_SAFETY_BALANCED = "0.87"
$env:ROUTER_SAFETY_PREMIUM = "0.85"
$py = "C:\Users\012\SKT LLM\.venv\Scripts\python.exe"
$res = "$root\src\ossp_router\resources"
$art = "$res\learned-router.v1.json"

Copy-Item "$art" "$art.e27.bak" -Force
Copy-Item "$res\learned-router-heavy.v1.json" "$res\learned-router-heavy.v1.json.e27.bak" -Force
Write-Host "[deploy-e43] backups written (*.e27.bak)"

Write-Host "[deploy-e43] 1) linear head (Train+Dev, alpha 10, legacy blend 0.9)"
& $py tools/train_learned_router_gpu.py `
  --input data/combined/inputs.json --outcomes data/combined/outcomes.json `
  --validation-input data/materialized/dev/inputs.json --validation-outcomes data/dev/outcomes.json `
  --artifact $art --report reports/learned-router-gpu-report.v1.json `
  --word-bins 8192 --char-bins 8192 --alphas 10 --blend-weights 0.9 --context-limits 1000000
if ($LASTEXITCODE -ne 0) { throw "linear stage failed" }

Write-Host "[deploy-e43] 2) augmentation (family 0.15, kNN conf 0.25)"
& $py tools/build_router_augmentation.py --artifact $art `
  --train-input data/combined/inputs.json --train-outcomes data/combined/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "augmentation stage failed" }

Write-Host "[deploy-e43] 3) meta GBM (blend .6/.45/.3, rank beta .4, safety .98/.87/.85)"
& $py tools/build_meta_gbm.py --artifact $art `
  --train-input data/combined/inputs.json --train-outcomes data/combined/outcomes.json `
  --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
if ($LASTEXITCODE -ne 0) { throw "meta stage failed" }

Write-Host "[deploy-e43] 5) public lookup"
& $py tools/build_public_lookup.py --artifact $art --inputs data/materialized/train/inputs.json data/materialized/dev/inputs.json
if ($LASTEXITCODE -ne 0) { throw "lookup stage failed" }

Write-Host "[deploy-e43] 6) pack heavy block"
& $py tools/pack_artifact.py --artifact $art
if ($LASTEXITCODE -ne 0) { throw "pack stage failed" }

Write-Host "[deploy-e43] artifact sizes"
Get-ChildItem $res\*.json | Select-Object Name, Length | Format-Table -AutoSize
Write-Host "[deploy-e43] DONE"
