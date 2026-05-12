# 01_download_arshnoor.ps1
# Descarga arshnoor7389/garbage-classification-dataset (4,3 GB) en ./data/raw/arshnoor

$ErrorActionPreference = "Stop"
$root = "C:\Users\mitgar14\Documentos\embebidos-3"
$rawDir = "$root\data\raw"
$targetDir = "$rawDir\arshnoor"

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

# Verificar kaggle CLI
$kaggleCmd = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggleCmd) {
    Write-Host "[!] kaggle CLI no instalado. Instala con: uv tool install kaggle" -ForegroundColor Yellow
    Write-Host "    Despues coloca tu kaggle.json en %USERPROFILE%\.kaggle\kaggle.json"
    exit 1
}

Set-Location $rawDir
Write-Host "[+] Descargando dataset (4,3 GB)..."
kaggle datasets download arshnoor7389/garbage-classification-dataset -p .
Write-Host "[+] Extrayendo..."
Expand-Archive -Path "garbage-classification-dataset.zip" -DestinationPath $targetDir -Force
Remove-Item "garbage-classification-dataset.zip"
Write-Host "[OK] Dataset listo en $targetDir"
Write-Host "    Verificar: cat $targetDir\Dataset\data.yaml"
