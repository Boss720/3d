param(
    [string]$PackageDir = (Join-Path $PSScriptRoot 'dist'),
    [string]$AppVersion = '0.2.0-beta.6'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$iscc = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
if (-not (Test-Path -LiteralPath $iscc -PathType Leaf)) {
    throw "Inno Setup compiler was not found at $iscc. Install Inno Setup 6 to build the installer."
}

$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$exe = Join-Path $PackageDir 'StereoLab.exe'
$license = Join-Path $PackageDir 'LICENSE'
$launcher = Join-Path $PackageDir 'Start-StereoLab.ps1'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw 'Build dist\StereoLab.exe first by running .\build-exe.ps1.'
}
if (-not (Test-Path -LiteralPath $license -PathType Leaf)) {
    throw 'The companion dist\LICENSE file is missing.'
}
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "The selected package is missing Start-StereoLab.ps1. Rebuild it with .\build-exe.ps1."
}

$output = Join-Path $PSScriptRoot '.runtime\stereolab-installer'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$appVersionArg = '/DAppVersion=' + $AppVersion
$packageDirArg = '/DPackageDir=' + $PackageDir
& $iscc $appVersionArg $packageDirArg (Join-Path $PSScriptRoot 'StereoLab.iss')
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE."
}

Write-Host "Built installer candidate in $output"
