$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$gpuPython = Join-Path $PSScriptRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $gpuPython)) {
    throw "DirectML environment not found. Run .\setup-gpu.ps1 first."
}

& $gpuPython -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $gpuPython -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller." }
}

& $gpuPython -m PyInstaller --clean --noconfirm StereoLab.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed to build StereoLab.exe." }

Write-Host "Built $PSScriptRoot\dist\StereoLab.exe"
