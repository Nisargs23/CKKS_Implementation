# Automated OpenFHE Build Script for Windows
# This script builds OpenFHE using MSYS2/MinGW-w64 GCC

param(
    [ValidateSet("lowmem", "balanced", "fast")]
    [string]$BuildProfile = "balanced",

    [int]$Jobs = 0
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  OpenFHE Build Script for Windows (MSYS2)" -ForegroundColor Cyan  
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Paths
$msys2Path = "C:\msys64\mingw64\bin"
$openfheSource = "D:\Manipal\Research\openfhe-development"
$openfheBuild = "$openfheSource\build"
$openfheInstall = "D:\Manipal\Research\openfhe-install"

$cpuCount = [Environment]::ProcessorCount
$releaseFlags = "-O3 -DNDEBUG"
$defaultJobs = 4
$profileMessage = "Balanced profile: optimized runtime with moderate build parallelism."

switch ($BuildProfile) {
    "lowmem" {
        $releaseFlags = "-O2 -DNDEBUG"
        $defaultJobs = 2
        $profileMessage = "Low-memory profile: reduced optimization and lower parallelism."
    }
    "balanced" {
        $releaseFlags = "-O3 -DNDEBUG"
        $defaultJobs = [Math]::Min([Math]::Max($cpuCount / 2, 2), 6)
        $profileMessage = "Balanced profile: optimized runtime with moderate build parallelism."
    }
    "fast" {
        $releaseFlags = "-O3 -march=native -mtune=native -DNDEBUG"
        $defaultJobs = [Math]::Max($cpuCount - 1, 2)
        $profileMessage = "Fast profile: host-native optimization and higher parallelism."
    }
}

$buildJobs = if ($Jobs -gt 0) { $Jobs } else { [int]$defaultJobs }

# Check MSYS2 installation
if (-not (Test-Path $msys2Path)) {
    Write-Host "ERROR: MSYS2 not found at $msys2Path" -ForegroundColor Red
    Write-Host "Please wait for MSYS2 and GCC installation to complete" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using MSYS2 toolchain at: $msys2Path" -ForegroundColor Green

# Add MSYS2 to PATH for this session
$env:Path = "$msys2Path;$env:Path"

# Verify tools
Write-Host "`nVerifying tools..." -ForegroundColor Cyan
try {
    $gccVersion = & gcc --version 2>&1 | Select-Object -First 1
    Write-Host "GCC: $gccVersion" -ForegroundColor Green
    
    $cmakeVersion = & cmake --version 2>&1 | Select-Object -First 1
    Write-Host "CMake: $cmakeVersion" -ForegroundColor Green
    
    $ninjaVersion = & ninja --version 2>&1
    Write-Host "Ninja: $ninjaVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Required tools not found" -ForegroundColor Red
    exit 1
}

# Clean and create build directory
Write-Host "`nPreparing build directory..." -ForegroundColor Cyan
if (Test-Path $openfheBuild) {
    Remove-Item -Recurse -Force $openfheBuild
}
New-Item -ItemType Directory -Path $openfheBuild | Out-Null
Set-Location $openfheBuild

# Configure with CMake
Write-Host "`nConfiguring OpenFHE with CMake..." -ForegroundColor Cyan
Write-Host $profileMessage -ForegroundColor Yellow
Write-Host "Release flags: $releaseFlags" -ForegroundColor Yellow
Write-Host "Build jobs: $buildJobs" -ForegroundColor Yellow
cmake .. `
    -G "Ninja" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$openfheInstall" `
    -DBUILD_UNITTESTS=OFF `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_BENCHMARKS=OFF `
    -DCMAKE_C_COMPILER=gcc `
    -DCMAKE_CXX_COMPILER=g++ `
    -DCMAKE_CXX_FLAGS_RELEASE="$releaseFlags"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: CMake configuration failed!" -ForegroundColor Red
    Set-Location "D:\Manipal\Research\Implementation"
    exit 1
}

# Build
Write-Host "`nBuilding OpenFHE (this will take 25-35 minutes)..." -ForegroundColor Cyan
Write-Host "Building with profile '$BuildProfile' using -j$buildJobs" -ForegroundColor Yellow
$ninjaBuildArgs = @("-j", "$buildJobs")
ninja @ninjaBuildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Build failed!" -ForegroundColor Red
    Set-Location "D:\Manipal\Research\Implementation"
    exit 1
}

# Install
Write-Host "`nInstalling OpenFHE..." -ForegroundColor Cyan
ninja install

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: Installation failed!" -ForegroundColor Red
    Set-Location "D:\Manipal\Research\Implementation"
    exit 1
}

Set-Location "D:\Manipal\Research\Implementation"

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  OpenFHE Build Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "`nInstalled to: $openfheInstall" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Build your project:" -ForegroundColor Yellow
Write-Host "   cd D:\Manipal\Research\Implementation" -ForegroundColor White
Write-Host "   .\build-with-msys2.ps1" -ForegroundColor White
Write-Host ""
