# Build Script for CKKS Bootstrapping Project (using MSYS2)

param(
    [ValidateSet("root", "final")]
    [string]$Project = "root",
    [switch]$KeepBuildDir
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  CKKS Bootstrapping Project Build (MSYS2)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Paths
$msys2Path = "C:\msys64\mingw64\bin"
$openfheInstall = "D:\Manipal\Research\openfhe-install"
$projectRoot = "D:\Manipal\Research\Implementation"
$gccPath = "$msys2Path\gcc.exe"
$gxxPath = "$msys2Path\g++.exe"

if ($Project -eq "final") {
    $sourceDir = "$projectRoot\final"
    $buildDir = "$projectRoot\final\build_msys2"
} else {
    $sourceDir = $projectRoot
    $buildDir = "$projectRoot\build"
}

# Check MSYS2
if (-not (Test-Path $msys2Path)) {
    Write-Host "ERROR: MSYS2 not found" -ForegroundColor Red
    exit 1
}

# Check OpenFHE
if (-not (Test-Path "$openfheInstall\include\openfhe")) {
    Write-Host "ERROR: OpenFHE not found at $openfheInstall" -ForegroundColor Red
    Write-Host "Please run .\build-openfhe.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Add MSYS2 to PATH
$env:Path = "$openfheInstall\lib;$msys2Path;$env:Path"

Write-Host "Using MSYS2 toolchain" -ForegroundColor Green
Write-Host "OpenFHE location: $openfheInstall" -ForegroundColor Green
Write-Host "Project source: $sourceDir" -ForegroundColor Green
Write-Host "Build directory: $buildDir" -ForegroundColor Green

# Clean and create build directory
Write-Host "`nPreparing build directory..." -ForegroundColor Cyan
if ((Test-Path $buildDir) -and -not $KeepBuildDir) {
    Remove-Item -Recurse -Force $buildDir
}
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}
Set-Location $buildDir

# Configure
Write-Host "`nConfiguring project with CMake..." -ForegroundColor Cyan
cmake $sourceDir `
    -G "Ninja" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="$openfheInstall" `
    -DOpenFHE_DIR="$openfheInstall/CMake" `
    -DCMAKE_C_COMPILER="$gccPath" `
    -DCMAKE_CXX_COMPILER="$gxxPath"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: CMake configuration failed!" -ForegroundColor Red
    Set-Location $projectRoot
    exit 1
}

# Build
Write-Host "`nBuilding project..." -ForegroundColor Cyan
ninja

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Build failed!" -ForegroundColor Red
    Set-Location $projectRoot
    exit 1
}

Set-Location $projectRoot

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "`nTo run the program:" -ForegroundColor Cyan
if ($Project -eq "final") {
    Write-Host "  cd final\build_msys2\bin" -ForegroundColor Yellow
} else {
    Write-Host "  cd build\bin" -ForegroundColor Yellow
}
Write-Host "  `$env:Path = `"D:\Manipal\Research\openfhe-install\lib;C:\msys64\mingw64\bin;`$env:Path`"" -ForegroundColor Yellow
Write-Host "  .\ckks_bootstrap_image.exe" -ForegroundColor Yellow
Write-Host ""
