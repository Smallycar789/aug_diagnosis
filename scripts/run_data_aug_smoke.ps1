# 数据增强冒烟测试（envv 环境）
# 用法: conda activate envv; powershell -File scripts/run_data_aug_smoke.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$py = "C:\Users\86188\miniconda3\envs\envv\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$configs = @(
    "configs/data_aug/smoke_vae_cooler.yaml",
    "configs/data_aug/smoke_gan_cooler.yaml",
    "configs/data_aug/smoke_gan_vae_sifuqi.yaml",
    "configs/data_aug/smoke_rvae_cooler.yaml"
)

foreach ($cfg in $configs) {
    Write-Host "`n========== $cfg ==========" -ForegroundColor Cyan
    & $py data_aug/train.py --config $cfg --stage all
}

Write-Host "`nAll data_aug smoke tests passed." -ForegroundColor Green
