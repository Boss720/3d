$ErrorActionPreference = 'Stop'
$cudaEnvironment = Join-Path $PSScriptRoot '.venv-cuda'
$cudaPython = Join-Path $cudaEnvironment 'Scripts/python.exe'

if (-not (Test-Path -LiteralPath $cudaPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $cudaEnvironment
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $cudaEnvironment
    } else {
        throw 'Python 3.10 or newer is required.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the CUDA environment.' }
}

& $cudaPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-cuda.txt')
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime installation failed. Other StereoLab environments are unchanged.' }
& $cudaPython -c "import onnxruntime as ort; p=ort.get_available_providers(); print('Runtime providers:', p); raise SystemExit(0 if 'CUDAExecutionProvider' in p else 1)"
if ($LASTEXITCODE -ne 0) { throw 'The installed runtime does not expose CUDAExecutionProvider.' }
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi --query-gpu=index,name,driver_model.current,memory.total --format=csv
    if ($LASTEXITCODE -ne 0) { Write-Warning 'nvidia-smi could not report device and driver mode. CUDA inference will confirm availability.' }
} else {
    Write-Warning 'nvidia-smi was not found. Confirm that an NVIDIA driver is installed before running CUDA inference.'
}
Write-Host 'CUDA runtime installed. Start it with .\run-cuda.ps1; the first named-model inference verifies GPU execution.'
