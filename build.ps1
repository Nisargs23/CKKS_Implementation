# Build Script for Windows

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  CKKS Bootstrapping Build Script (Windows)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if CMake is installed
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "ERROR: CMake is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install CMake from https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

Write-Host "CMake found: $($cmake.Source)" -ForegroundColor Green

# Create build directory
$buildDir = "build"
if (Test-Path $buildDir) {
    Write-Host "Removing existing build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}

Write-Host "Creating build directory..." -ForegroundColor Cyan
New-Item -ItemType Directory -Path $buildDir | Out-Null
Set-Location $buildDir

# Configure
Write-Host ""
Write-Host "Configuring project with CMake..." -ForegroundColor Cyan
Write-Host "NOTE: If using vcpkg, make sure to set the toolchain file" -ForegroundColor Yellow

# Try to find vcpkg automatically
$vcpkgPath = $env:VCPKG_ROOT
if ($vcpkgPath -and (Test-Path "$vcpkgPath\scripts\buildsystems\vcpkg.cmake")) {
    Write-Host "Found vcpkg at: $vcpkgPath" -ForegroundColor Green
    cmake .. -DCMAKE_TOOLCHAIN_FILE="$vcpkgPath\scripts\buildsystems\vcpkg.cmake"
} else {
    Write-Host "vcpkg not found in VCPKG_ROOT environment variable" -ForegroundColor Yellow
    Write-Host "Attempting standard CMake configuration..." -ForegroundColor Yellow
    cmake ..
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    Write-Host "Please ensure OpenFHE is installed and accessible" -ForegroundColor Yellow
    Set-Location ..
    exit 1
}

# Build
Write-Host ""
Write-Host "Building project..." -ForegroundColor Cyan
cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location ..

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Build completed successfully!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the program:" -ForegroundColor Cyan
Write-Host "  cd build\bin\Release" -ForegroundColor Yellow
Write-Host "  .\ckks_bootstrap_image.exe" -ForegroundColor Yellow
Write-Host ""
