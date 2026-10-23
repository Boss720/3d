$ErrorActionPreference = 'Stop'
$cudaPython = Join-Path $PSScriptRoot '.venv-cuda/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $cudaPython)) {
    throw 'CUDA environment is missing. Run .\setup-cuda.ps1 first.'
}
& $cudaPython (Join-Path $PSScriptRoot 'main.py')
exit $LASTEXITCODE
