$ErrorActionPreference = 'Stop'
$cudaEnvironment = Join-Path $PSScriptRoot '.venv-cuda'
$cudaPython = Join-Path $cudaEnvironment 'Scripts/python.exe'
$setupTemp = Join-Path $PSScriptRoot '.runtime/cuda-setup-temp'
New-Item -ItemType Directory -Path $setupTemp -Force | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$env:TEMP = $setupTemp
$env:TMP = $setupTemp

try {
    $environmentReady = $false
    if (Test-Path -LiteralPath $cudaPython -PathType Leaf) {
        & $cudaPython -c "import pip,sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" 2>$null
        $environmentReady = $LASTEXITCODE -eq 0
    }

    if (-not $environmentReady) {
        $pythonCommand = $null
        $pythonSelector = @()
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 --version 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { $pythonSelector = @('-3.12') }
            else { $pythonSelector = @('-3') }
            $pythonCommand = 'py'
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            $pythonCommand = 'python'
        } else {
            throw 'Python 3.10 or newer is required.'
        }

        $venvArguments = @('-m', 'venv')
        if (Test-Path -LiteralPath $cudaEnvironment) { $venvArguments += '--clear' }
        $venvArguments += $cudaEnvironment
        & $pythonCommand @pythonSelector @venvArguments
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
    Write-Host 'CUDA runtime installed. Start with .\run-cuda.ps1 or build the packaged app with .\build-cuda-exe.ps1.'
}
finally {
    if ($null -eq $previousTemp) { Remove-Item Env:TEMP -ErrorAction SilentlyContinue } else { $env:TEMP = $previousTemp }
    if ($null -eq $previousTmp) { Remove-Item Env:TMP -ErrorAction SilentlyContinue } else { $env:TMP = $previousTmp }
}
