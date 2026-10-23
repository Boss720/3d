param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'),
    [string]$DistPath = (Join-Path $PSScriptRoot 'dist'),
    [string]$WorkPath = (Join-Path $PSScriptRoot 'build'),
    [ValidateSet('directml', 'cuda')]
    [string]$Runtime = 'directml'
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Build Python not found at $PythonPath. Run .\setup-gpu.ps1 or pass -PythonPath."
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$DistPath = [System.IO.Path]::GetFullPath($DistPath)
$WorkPath = [System.IO.Path]::GetFullPath($WorkPath)

if ($Runtime -eq 'cuda') {
    & $PythonPath -c "import onnxruntime as ort; p=ort.get_available_providers(); print('Runtime providers:', p); raise SystemExit(0 if 'CUDAExecutionProvider' in p else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA build requires CUDAExecutionProvider in the selected Python environment.' }
}

& $PythonPath -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $PythonPath -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller." }
}

$previousCudaBundle = $env:STEREOLAB_CUDA_BUNDLE
try {
    if ($Runtime -eq 'cuda') { $env:STEREOLAB_CUDA_BUNDLE = '1' }
    else { Remove-Item Env:STEREOLAB_CUDA_BUNDLE -ErrorAction SilentlyContinue }
    & $PythonPath -m PyInstaller --clean --noconfirm --distpath $DistPath --workpath $WorkPath StereoLab.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed to build StereoLab.exe." }
}
finally {
    if ($null -eq $previousCudaBundle) { Remove-Item Env:STEREOLAB_CUDA_BUNDLE -ErrorAction SilentlyContinue }
    else { $env:STEREOLAB_CUDA_BUNDLE = $previousCudaBundle }
}

$licenseSource = Join-Path $PSScriptRoot "LICENSE"
$packagePath = if ($Runtime -eq 'cuda') { Join-Path $DistPath 'StereoLab' } else { $DistPath }
$licenseDestination = Join-Path $packagePath "LICENSE"
New-Item -ItemType Directory -Path $packagePath -Force | Out-Null
Copy-Item -LiteralPath $licenseSource -Destination $licenseDestination -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Start-StereoLab.ps1') -Destination $packagePath -Force

Write-Host "Built $(Join-Path $packagePath 'StereoLab.exe') ($Runtime runtime)"
Write-Host "Copied $licenseDestination"
