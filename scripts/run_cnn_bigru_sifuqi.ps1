$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = "C:\Users\86188\miniconda3\envs\envv\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "=== CNN-BiGRU fault diagnosis on sifuqi ===" -ForegroundColor Cyan
& $Python diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training finished." -ForegroundColor Green
