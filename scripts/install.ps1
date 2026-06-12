# FastVideoEdit one-liner installer.
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/Rxd-essss/FastVideoEdit/main/scripts/install.ps1 | iex
# Downloads the latest main into .\FastVideoEdit and starts the editor
# (run.ps1 creates the venv and installs dependencies on first run).
$ErrorActionPreference = 'Stop'

$repo = 'Rxd-essss/FastVideoEdit'
$dest = Join-Path (Get-Location) 'FastVideoEdit'
if (Test-Path $dest) {
    Write-Host "Папка уже существует: $dest" -ForegroundColor Yellow
    Write-Host 'Удалите/переименуйте её или запустите установку из другой папки.'
    return
}

# Windows PowerShell 5.1 defaults to TLS 1.0 — GitHub requires 1.2+.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

Write-Host "Скачиваю $repo ..." -ForegroundColor Cyan
$zip = Join-Path $env:TEMP 'FastVideoEdit-main.zip'
Invoke-WebRequest "https://github.com/$repo/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing

$tmp = Join-Path $env:TEMP ('fve_install_' + [IO.Path]::GetRandomFileName())
Expand-Archive $zip -DestinationPath $tmp
Move-Item (Join-Path $tmp 'FastVideoEdit-main') $dest
Remove-Item $zip -Force
Remove-Item $tmp -Recurse -Force

Write-Host "Установлено в $dest" -ForegroundColor Green
Write-Host 'Первый запуск создаст .venv и поставит зависимости (несколько минут).' -ForegroundColor Cyan
Set-Location $dest
powershell -ExecutionPolicy Bypass -File .\run.ps1
