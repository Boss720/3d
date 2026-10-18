$ErrorActionPreference = 'Stop'
$gpuEnvironment = Join-Path $PSScriptRoot '.venv-gpu'
$gpuPython = Join-Path $gpuEnvironment 'Scripts/python.exe'

if (-not (Test-Path -LiteralPath $gpuPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $gpuEnvironment
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $gpuEnvironment
    } else {
        throw 'Python 3.10 or newer is required.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the GPU environment.' }
}

& $gpuPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-gpu.txt')
if ($LASTEXITCODE -ne 0) { throw 'GPU runtime installation failed. The CPU environment is still available.' }
& $gpuPython -c "import onnxruntime as ort; print('Runtime providers:', ort.get_available_providers()); raise SystemExit(0 if 'DmlExecutionProvider' in ort.get_available_providers() else 1)"
if ($LASTEXITCODE -ne 0) { throw 'The installed runtime does not expose DirectML.' }
Write-Host 'GPU runtime installed. Restart StereoLab with .\run.ps1. First inference verifies whether GPU execution works.'
