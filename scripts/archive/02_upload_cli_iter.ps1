# 02_upload_cli_iter.ps1
# Itera todas las imágenes de Dataset/ y las sube via roboflow CLI con su .txt de anotación
# Uso: .\scripts\02_upload_cli_iter.ps1

$ErrorActionPreference = "Continue"   # no abortar al primer fallo
$datasetDir = "C:\Users\mitgar14\Documentos\embebidos-3\data\raw\arshnoor\Dataset"
$projectId  = "waste-3class-lwld8"
$batchName  = "arshnoor-base"
$concurrency = 15   # subir en paralelo (PowerShell 7+ requerido para -Parallel)
$dataYaml   = "$datasetDir\data.yaml"

if (-not (Test-Path $dataYaml)) {
    Write-Error "No se encuentra $dataYaml"
    exit 1
}

$images = Get-ChildItem "$datasetDir\images" -Filter *.jpg
Write-Host "[+] $($images.Count) imágenes a subir → $projectId, batch=$batchName"

$start = Get-Date
$counter = [System.Threading.Thread]::CurrentThread.ManagedThreadId
$total = $images.Count

# PowerShell 7+ con -Parallel
$images | ForEach-Object -Parallel {
    $img = $_.FullName
    $txt = $_.FullName -replace '\\images\\', '\labels\' -replace '\.jpg$', '.txt'
    if (Test-Path $txt) {
        & uv run roboflow upload `
          --project $using:projectId `
          --annotation $txt `
          --labelmap $using:dataYaml `
          --batch $using:batchName `
          --retries 2 `
          $img 2>&1 | Out-Null
    } else {
        Write-Host "[!] sin label: $img"
    }
} -ThrottleLimit $concurrency

$elapsed = (Get-Date) - $start
Write-Host "[OK] Upload completo en $($elapsed.ToString('hh\:mm\:ss'))"
