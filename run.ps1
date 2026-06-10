# FastVideoEdit - launcher for the web editor.
# Creates the .venv and installs dependencies on first run, then starts serve.py.
# Usage:  .\run.ps1                 (open the editor, pick a clip in the UI)
#         .\run.ps1 --video x.mp4   (open a specific clip)
#         .\run.ps1 --port 8001     (use another port if 8000 is busy)
# If PowerShell blocks scripts:  powershell -ExecutionPolicy Bypass -File .\run.ps1
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
$sentinel = Join-Path $venv '.fve_installed'   # written only after a full install

if (-not (Test-Path $py)) {
    Write-Host 'First run: creating .venv ...' -ForegroundColor Cyan
    $base = $null
    foreach ($cand in @('py -3.12', 'py -3.11', 'py', 'python')) {
        $exe = $cand.Split(' ')[0]
        if (Get-Command $exe -ErrorAction SilentlyContinue) { $base = $cand; break }
    }
    if (-not $base) { Write-Error 'Python not found. Install Python 3.12 from https://python.org'; exit 1 }
    $parts = $base.Split(' '); $exe = $parts[0]
    $pre = if ($parts.Count -gt 1) { $parts[1..($parts.Count - 1)] } else { @() }
    Write-Host "  using: $base"
    & $exe @pre -m venv $venv
}

# Install deps unless a previous run completed successfully (sentinel present).
if (-not (Test-Path $sentinel)) {
    Write-Host '  installing dependencies (may take a couple of minutes) ...' -ForegroundColor Cyan
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'pip install failed - fix the error above and re-run (the venv is kept, install will retry).'
        exit 1
    }
    Set-Content -Path $sentinel -Value 'ok' -Encoding ascii
}

# ffmpeg is required at runtime; warn early (PATH may be stale right after install).
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) {
    $mp = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $up = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($mp + ';' + $up) -match 'ffmpeg|Gyan')) {
        Write-Host 'WARNING: ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg (then open a new terminal).' -ForegroundColor Yellow
    }
}

Write-Host 'Starting editor -> http://127.0.0.1:8000' -ForegroundColor Green
& $py (Join-Path $root 'serve.py') @args
