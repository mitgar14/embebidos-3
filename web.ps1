#!/usr/bin/env pwsh
# web.ps1 — Orquestación del frontend embebidos-3
# Uso: .\web.ps1 [-Action start|stop|restart|status] [-Dev]
# Requiere: pwsh 7+, bun en PATH
param(
  [ValidateSet('start','stop','restart','status')]
  [string]$Action = 'start',
  [switch]$Dev   # Si se pasa, usa "bun run dev" (HMR) en lugar de build+preview
)

$ErrorActionPreference = 'Stop'
$PidDir  = Join-Path $PSScriptRoot '.run'
$PidFile = Join-Path $PidDir 'web.pid'
$LogFile = Join-Path $PidDir 'web.log'
$WebDir  = Join-Path $PSScriptRoot 'web'
$NanoUrl = 'http://100.64.0.2:8000/health'
$PreviewPort = 4173
$DevPort     = 3000

# ─── Helpers ────────────────────────────────────────────────────────────────
function Ensure-RunDir {
  if (-not (Test-Path $PidDir)) { New-Item -ItemType Directory -Path $PidDir | Out-Null }
}

function Get-SavedPid {
  if (Test-Path $PidFile) {
    $id = Get-Content $PidFile -Raw
    return [int]($id.Trim())
  }
  return $null
}

function Is-ProcessAlive([int]$id) {
  try { $null = Get-Process -Id $id -ErrorAction Stop; return $true }
  catch { return $false }
}

function Kill-ProcessTree([int]$id) {
  if ($IsWindows) {
    # taskkill /T mata el árbol completo (bun → vite → esbuild)
    & taskkill /PID $id /T /F 2>$null
  } elseif ($IsMacOS) {
    # macOS: matar hijos primero, luego el padre
    & pkill -P $id 2>$null
    & kill $id 2>$null
  } else {
    # Linux: igual que macOS
    & pkill -P $id 2>$null
    & kill $id 2>$null
  }
}

function Open-Browser([string]$url) {
  if ($IsWindows) {
    Start-Process $url
  } elseif ($IsMacOS) {
    & open $url
  } else {
    & xdg-open $url 2>$null
  }
}

function Check-Nano {
  try {
    $r = Invoke-WebRequest -Uri $NanoUrl -TimeoutSec 3 -ErrorAction Stop
    if ($r.StatusCode -eq 200) {
      Write-Host "  Nano: OK ($NanoUrl)" -ForegroundColor Green
    }
  } catch {
    Write-Warning "  Nano no responde en $NanoUrl (la app igual arrancara)"
  }
}

function Ensure-Bun {
  if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Write-Error "bun no encontrado en PATH. Instalar desde https://bun.sh"
    exit 1
  }
}

# ─── Acciones ────────────────────────────────────────────────────────────────
function Action-Stop {
  $id = Get-SavedPid
  if (-not $id) { Write-Host "No hay proceso registrado en $PidFile"; return }
  if (Is-ProcessAlive $id) {
    Write-Host "Deteniendo proceso $id y sus hijos..."
    Kill-ProcessTree $id
    Write-Host "Detenido."
  } else {
    Write-Host "El proceso $id ya no estaba activo."
  }
  Remove-Item $PidFile -ErrorAction SilentlyContinue
}

function Action-Status {
  $id = Get-SavedPid
  if ($id -and (Is-ProcessAlive $id)) {
    Write-Host "Frontend ACTIVO (PID $id)"
  } else {
    Write-Host "Frontend NO está corriendo"
    if ($id) { Remove-Item $PidFile -ErrorAction SilentlyContinue }
  }
  Check-Nano
}

function Action-Start {
  Ensure-RunDir
  Ensure-Bun

  # Detener instancia anterior si existe
  $existing = Get-SavedPid
  if ($existing -and (Is-ProcessAlive $existing)) {
    Write-Host "Deteniendo instancia anterior (PID $existing)..."
    Kill-ProcessTree $existing
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
  }

  Push-Location $WebDir
  try {
    if ($Dev) {
      # ── Modo desarrollo (HMR) ──────────────────────────────────────────
      Write-Host "Instalando dependencias si hace falta..."
      & bun install
      Write-Host "Arrancando modo dev (puerto $DevPort)..."
      $proc = Start-Process -FilePath 'bun' `
                            -ArgumentList 'run','dev' `
                            -PassThru `
                            -RedirectStandardOutput $LogFile `
                            -RedirectStandardError  ($LogFile -replace '\.log$', '-err.log')
      $proc.Id | Out-File $PidFile -Encoding utf8
      Write-Host "Dev server arrancado (PID $($proc.Id))"
      Check-Nano
      Start-Sleep -Seconds 2  # darle tiempo al servidor para arrancar
      Open-Browser "http://localhost:$DevPort/#/"
    } else {
      # ── Modo producción / demo (build + preview) ───────────────────────
      Write-Host "Instalando dependencias si hace falta..."
      & bun install

      Write-Host "Construyendo..."
      & bun run build
      if ($LASTEXITCODE -ne 0) { Write-Error "Build fallido"; exit 1 }

      Write-Host "Levantando vite preview en puerto $PreviewPort..."
      $proc = Start-Process -FilePath 'bun' `
                            -ArgumentList 'run','preview' `
                            -PassThru `
                            -RedirectStandardOutput $LogFile `
                            -RedirectStandardError  ($LogFile -replace '\.log$', '-err.log')
      $proc.Id | Out-File $PidFile -Encoding utf8
      Write-Host "Preview arrancado (PID $($proc.Id))"
      Check-Nano
      Start-Sleep -Seconds 1
      Open-Browser "http://localhost:$PreviewPort/#/"
    }
  } finally {
    Pop-Location
  }
}

function Action-Restart {
  Action-Stop
  Start-Sleep -Milliseconds 500
  Action-Start
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────
switch ($Action) {
  'start'   { Action-Start }
  'stop'    { Action-Stop }
  'restart' { Action-Restart }
  'status'  { Action-Status }
}
