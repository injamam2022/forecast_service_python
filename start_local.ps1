# Start local Python forecast API using the project venv (has sklearn, etc.)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
  Write-Host 'Creating venv...'
  python -m venv .venv
  & $venvPy -m pip install -r requirements.txt
}

# Free port 8001
Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | ForEach-Object {
  Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1

$env:FORECAST_PORT = '8001'
$env:FORECAST_API_KEY = $env:FORECAST_API_KEY
Write-Host 'Starting forecast API on http://127.0.0.1:8001 ...'
& $venvPy -m uvicorn main:app --host 127.0.0.1 --port 8001
