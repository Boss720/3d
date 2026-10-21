param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'),
    [string]$DistPath = (Join-Path $PSScriptRoot 'dist'),
    [string]$WorkPath = (Join-Path $PSScriptRoot 'build')
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Build Python not found at $PythonPath. Run .\setup-gpu.ps1 or pass -PythonPath."
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$DistPath = [System.IO.Path]::GetFullPath($DistPath)
$WorkPath = [System.IO.Path]::GetFullPath($WorkPath)

& $PythonPath -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $PythonPath -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyInstaller." }
}

& $PythonPath -m PyInstaller --clean --noconfirm --distpath $DistPath --workpath $WorkPath StereoLab.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed to build StereoLab.exe." }

$licenseSource = Join-Path $PSScriptRoot "LICENSE"
$licenseDestination = Join-Path $DistPath "LICENSE"
Copy-Item -LiteralPath $licenseSource -Destination $licenseDestination -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Start-StereoLab.ps1') -Destination $DistPath -Force

Write-Host "Built $DistPath\StereoLab.exe"
Write-Host "Copied $licenseDestination"
