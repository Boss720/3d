$ErrorActionPreference = "Stop"

$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$gpuPython = Join-Path $PSScriptRoot '.venv-gpu/Scripts/python.exe'
if (Test-Path -LiteralPath $gpuPython) {
    & $gpuPython (Join-Path $PSScriptRoot "main.py")
} elseif (Test-Path -LiteralPath $projectPython) {
    & $projectPython (Join-Path $PSScriptRoot "main.py")
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 (Join-Path $PSScriptRoot "main.py")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python (Join-Path $PSScriptRoot "main.py")
} else {
    throw "Python 3 was not found. Install Python 3.10 or newer and try again."
}
