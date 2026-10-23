param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '.runtime\windows-cuda\dist\StereoLab'),
    [string]$AppVersion = '0.2.0-beta.5'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$iscc = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
if (-not (Test-Path -LiteralPath $iscc -PathType Leaf)) {
    throw "Inno Setup compiler was not found at $iscc. Install Inno Setup 6 to build the CUDA installer."
}

$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
foreach ($name in @('StereoLab.exe', 'LICENSE', 'Start-StereoLab.ps1')) {
    if (-not (Test-Path -LiteralPath (Join-Path $PackageDir $name) -PathType Leaf)) {
        throw "The CUDA package is missing $name. Run .\build-cuda-exe.ps1 first."
    }
}

$output = Join-Path $PSScriptRoot '.runtime\windows-cuda\installer'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$appVersionArg = '/DAppVersion=' + $AppVersion
$packageDirArg = '/DPackageDir=' + $PackageDir
& $iscc $appVersionArg $packageDirArg (Join-Path $PSScriptRoot 'StereoLab-CUDA.iss')
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE." }

Write-Host "Built CUDA installer candidate in $output"
