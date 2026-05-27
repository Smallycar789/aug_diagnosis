$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = "C:\Users\86188\miniconda3\envs\envv\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$configs = @(
    "configs/diagnosis/smoke_cnn_bigru_cwru.yaml",
    "configs/diagnosis/smoke_resnet_cwru.yaml",
    "configs/diagnosis/smoke_tl_meta_cwru.yaml"
)

foreach ($cfg in $configs) {
    Write-Host "=== Training: $cfg ===" -ForegroundColor Cyan
    & $Python diagnosis/train.py --config $cfg
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "All diagnosis smoke tests passed." -ForegroundColor Green
