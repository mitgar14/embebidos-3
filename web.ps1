#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Orquesta el frontend "Tiny Trash" (Vite) de embebidos-3.
.DESCRIPTION
  Levanta, detiene, reinicia o consulta el estado del frontend. Por defecto
  hace build + vite preview (:4173); con -Dev usa HMR (:3000). Registra el
  proceso en .run/web.pid y deja logs en .run/ (web.log = stdout del server,
  orchestrator.log = logs de este script).
.PARAMETER Action
  start | stop | restart | status. Default: start.
.PARAMETER Dev
  Usa "bun run dev" (HMR, puerto 3000) en vez de build + preview.
.EXAMPLE
  .\web.ps1
  Build + preview en http://localhost:4173
.EXAMPLE
  .\web.ps1 -Action restart
.EXAMPLE
  .\web.ps1 -Action stop -WhatIf
  Muestra qué procesos detendría, sin matarlos.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
param(
  [ValidateSet('start','stop','restart','status')]
  [string]$Action = 'start',
  [switch]$Dev
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Este script maneja $LASTEXITCODE a mano (bun build) y tolera códigos != 0
# esperados (taskkill/pkill sobre procesos ya muertos): no auto-lanzar.
$PSNativeCommandUseErrorActionPreference = $false

# ─── Configuración ──────────────────────────────────────────────────────────
$PidDir      = Join-Path $PSScriptRoot '.run'
$PidFile     = Join-Path $PidDir 'web.pid'
$LogFile     = Join-Path $PidDir 'web.log'           # stdout/err del server Vite
$RunLog      = Join-Path $PidDir 'orchestrator.log'  # logs de este script
$WebDir      = Join-Path $PSScriptRoot 'web'
$NanoUrl     = 'http://100.64.0.2:8000/health'
$PreviewPort = 4173
$DevPort     = 3000

# ─── Logging ──────────────────────────────────────────────────────────────────
# Niveles INFO/OK/WARN/ERROR con timestamp y color via $PSStyle (PS 7.2+).
# Degrada a texto plano si: PS < 7.2, NO_COLOR, o salida redirigida
# ($PSStyle.OutputRendering='Host' elimina ANSI al pipear/redirigir).
function Write-Log {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)][ValidateSet('INFO','OK','WARN','ERROR')][string]$Level,
    [Parameter(Mandatory)][string]$Message
  )
  $stamp = Get-Date -Format 'HH:mm:ss'
  $tag = switch ($Level) {
    'INFO'  { '[INFO]' }
    'OK'    { '[ OK ]' }
    'WARN'  { '[WARN]' }
    'ERROR' { '[FAIL]' }
  }

  $useColor = ($PSVersionTable.PSVersion -ge [version]'7.2') -and (-not $env:NO_COLOR)
  if ($useColor) {
    $color = switch ($Level) {
      'INFO'  { $PSStyle.Foreground.BrightCyan }
      'OK'    { $PSStyle.Foreground.BrightGreen }
      'WARN'  { $PSStyle.Foreground.BrightYellow }
      'ERROR' { $PSStyle.Foreground.BrightRed }
    }
    $dim = $PSStyle.Foreground.BrightBlack
    $rst = $PSStyle.Reset
    Write-Host "$dim$stamp$rst $color$tag$rst $Message"
  } else {
    Write-Host "$stamp $tag $Message"
  }

  # Copia a archivo en texto plano (para revisar tras un fallo).
  $plain = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $tag $Message"
  Add-Content -Path $RunLog -Value $plain -Encoding utf8 -ErrorAction SilentlyContinue
}

# ─── Helpers ────────────────────────────────────────────────────────────────
function Initialize-RunDir {
  if (-not (Test-Path $PidDir)) {
    New-Item -ItemType Directory -Path $PidDir | Out-Null
  }
}

function Get-SavedPid {
  if (-not (Test-Path $PidFile)) { return $null }
  $raw = (Get-Content $PidFile -Raw).Trim()
  if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
  $parsed = 0
  if ([int]::TryParse($raw, [ref]$parsed)) { return $parsed }
  Write-Verbose "PID file con contenido inválido: '$raw'"
  return $null
}

function Test-ProcessAlive([int]$Id) {
  try { $null = Get-Process -Id $Id -ErrorAction Stop; return $true }
  catch { return $false }
}

function Stop-ProcessTree {
  [CmdletBinding(SupportsShouldProcess)]
  param([Parameter(Mandatory)][int]$Id)

  if (-not $PSCmdlet.ShouldProcess("PID $Id y su árbol de procesos hijos", 'Detener')) {
    return
  }

  if ($IsWindows) {
    # taskkill /T mata el árbol completo (bun -> vite -> esbuild)
    $out = & taskkill /PID $Id /T /F 2>&1
    Write-Verbose "taskkill: $out"
  } else {
    # macOS / Linux: hijos primero, luego el padre
    $out = & pkill -P $Id 2>&1; Write-Verbose "pkill: $out"
    $out = & kill $Id 2>&1;      Write-Verbose "kill: $out"
  }

  # pkill es asíncrono: esperar (hasta 5s) a que el proceso muera de verdad.
  $deadline = (Get-Date).AddSeconds(5)
  while ((Test-ProcessAlive $Id) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 150
  }
}

function Open-Browser([string]$Url) {
  if     ($IsWindows) { Start-Process $Url }
  elseif ($IsMacOS)   { & open $Url }
  else                { & xdg-open $Url 2>&1 | Out-Null }
}

function Wait-HttpReady([int]$Port, [int]$TimeoutSec = 20) {
  # vite preview suele escuchar solo en IPv6 (::1); sondear "localhost" daba
  # falsos negativos en Windows (resolvía a 127.0.0.1 y agotaba el timeout sin
  # caer a ::1). Sondeamos ambas loopback explícitas; primera que responda gana.
  $urls = @("http://[::1]:$Port/", "http://127.0.0.1:$Port/")
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    foreach ($u in $urls) {
      try {
        $r = Invoke-WebRequest -Uri $u -TimeoutSec 1 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { return $true }
      } catch { }
    }
    Start-Sleep -Milliseconds 200
  }
  return $false
}

function Test-NanoHealth {
  try {
    $r = Invoke-WebRequest -Uri $NanoUrl -TimeoutSec 3 -ErrorAction Stop
    if ($r.StatusCode -eq 200) {
      Write-Log OK "Nano responde ($NanoUrl)"
    } else {
      Write-Log WARN "Nano respondió HTTP $($r.StatusCode) en $NanoUrl"
    }
  } catch {
    Write-Log WARN "Nano no responde en $NanoUrl (la app igual arranca)"
  }
}

function Assert-Bun {
  if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Write-Log ERROR "bun no encontrado en PATH. Instalar desde https://bun.sh"
    exit 2
  }
}

# ─── Acciones ────────────────────────────────────────────────────────────────
function Invoke-StopAction {
  $id = Get-SavedPid
  if (-not $id) {
    Write-Log INFO "No hay proceso registrado en $PidFile"
    return
  }
  if (Test-ProcessAlive $id) {
    Write-Log INFO "Deteniendo proceso $id y sus hijos..."
    Stop-ProcessTree -Id $id
    if (Test-ProcessAlive $id) {
      Write-Log WARN "El proceso $id sigue vivo tras el intento de kill"
    } else {
      Write-Log OK "Detenido (PID $id)"
    }
  } else {
    Write-Log INFO "El proceso $id ya no estaba activo"
  }
  Remove-Item $PidFile -ErrorAction SilentlyContinue
}

function Invoke-StatusAction {
  $id = Get-SavedPid
  if ($id -and (Test-ProcessAlive $id)) {
    Write-Log OK "Frontend ACTIVO (PID $id)"
  } else {
    Write-Log INFO "Frontend NO está corriendo"
    if ($id) { Remove-Item $PidFile -ErrorAction SilentlyContinue }
  }
  Test-NanoHealth
}

function Invoke-StartAction {
  Initialize-RunDir
  Assert-Bun

  $existing = Get-SavedPid
  if ($existing -and (Test-ProcessAlive $existing)) {
    Write-Log INFO "Deteniendo instancia anterior (PID $existing)..."
    Stop-ProcessTree -Id $existing
    Remove-Item $PidFile -ErrorAction SilentlyContinue
  }

  $port    = if ($Dev) { $DevPort } else { $PreviewPort }
  $bunArgs = if ($Dev) { @('run','dev') } else { @('run','preview') }
  $mode    = if ($Dev) { 'dev (HMR)' } else { 'preview' }

  Push-Location $WebDir
  try {
    Write-Log INFO "Instalando dependencias (bun install)..."
    & bun install
    if ($LASTEXITCODE -ne 0) { Write-Log ERROR "bun install falló (exit $LASTEXITCODE)"; exit 1 }

    if (-not $Dev) {
      Write-Log INFO "Construyendo (bun run build)..."
      & bun run build
      if ($LASTEXITCODE -ne 0) { Write-Log ERROR "Build falló (exit $LASTEXITCODE)"; exit 1 }
      Write-Log OK "Build completo"
    }

    Write-Log INFO "Levantando $mode en puerto $port..."
    $errLog = $LogFile -replace '\.log$', '-err.log'
    $proc = Start-Process -FilePath 'bun' `
                          -ArgumentList $bunArgs `
                          -PassThru `
                          -RedirectStandardOutput $LogFile `
                          -RedirectStandardError  $errLog
    $proc.Id | Out-File $PidFile -Encoding utf8
    Write-Log OK "Servidor lanzado (PID $($proc.Id)); stdout en $LogFile"
  } finally {
    Pop-Location
  }

  $url = "http://localhost:$port/"
  if (Wait-HttpReady $port) {
    Write-Log OK "Servidor listo en $url"
  } else {
    Write-Log WARN "El servidor no respondió a tiempo; abriendo el navegador igual"
  }
  Test-NanoHealth
  Write-Log INFO "Abriendo navegador en ${url}#/"
  Open-Browser "${url}#/"
}

function Invoke-RestartAction {
  Invoke-StopAction
  Start-Sleep -Milliseconds 300
  Invoke-StartAction
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────
try {
  switch ($Action) {
    'start'   { Invoke-StartAction }
    'stop'    { Invoke-StopAction }
    'restart' { Invoke-RestartAction }
    'status'  { Invoke-StatusAction }
  }
  exit 0
}
catch {
  Write-Log ERROR $_.Exception.Message
  Write-Verbose $_.ScriptStackTrace
  exit 1
}
