param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv-cuda\Scripts\python.exe'),
    [string]$DistPath = (Join-Path $PSScriptRoot '.runtime\windows-cuda\dist'),
    [string]$WorkPath = (Join-Path $PSScriptRoot '.runtime\windows-cuda\build')
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'CUDA Python environment not found. Run .\setup-cuda.ps1 first.'
}

& (Join-Path $PSScriptRoot 'build-exe.ps1') -PythonPath $PythonPath -DistPath $DistPath -WorkPath $WorkPath -Runtime cuda
if ($LASTEXITCODE -ne 0) { throw 'CUDA executable build failed.' }

if (-not (Test-Path -LiteralPath (Join-Path $DistPath 'StereoLab\StereoLab.exe') -PathType Leaf)) {
    throw 'The CUDA build did not produce the expected one-folder package.'
}
