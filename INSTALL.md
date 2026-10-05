# Quick Start Installation Guide

## Current Status
✅ CMake installed
✅ Visual Studio Build Tools installing...
⏳ OpenFHE - ready to build

## Step-by-Step Instructions

### 1. Wait for Visual Studio Build Tools to Complete
The installation is currently in progress. It will take 5-10 minutes.

### 2. Build OpenFHE

Once Visual Studio finishes installing, follow these steps:

#### Open Visual Studio Developer Command Prompt:
1. Press `Win + S` and search for "**x64 Native Tools Command Prompt for VS 2022**"
2. Open it as Administrator (right-click → Run as administrator)

#### In that command prompt, run:
```cmd
cd D:\Manipal\Research\openfhe-development\build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX="D:/Manipal/Research/openfhe-install" -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF

cmake --build . --config Release -j8

cmake --install . --config Release
```

**Note:** Building will take 20-30 minutes. The `-j8` flag uses 8 CPU cores to speed it up.

### 3. Build Your Project

After OpenFHE is installed:

```cmd
cd D:\Manipal\Research\Implementation
mkdir build
cd build

cmake .. -DCMAKE_PREFIX_PATH="D:/Manipal/Research/openfhe-install" -G "Visual Studio 17 2022" -A x64

cmake --build . --config Release

cd bin\Release
ckks_bootstrap_image.exe
```

## Alternative: Use PowerShell with Developer Environment

If you prefer PowerShell, you can source the VS environment:

```powershell
# Import Visual Studio environment
$vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
if (Test-Path "$vsPath\Common7\Tools\Launch-VsDevShell.ps1") {
    & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
}

# Then run the same cmake commands as above
```

## Troubleshooting

### If CMake can't find the compiler:
Make sure you're using the **Visual Studio Developer Command Prompt**, not regular PowerShell/CMD.

### If build fails with memory errors:
Reduce parallel jobs: use `-j4` instead of `-j8`

### If OpenFHE not found:
Make sure CMAKE_PREFIX_PATH points to where you installed OpenFHE:
```
-DCMAKE_PREFIX_PATH="D:/Manipal/Research/openfhe-install"
```

## What Gets Installed

- **OpenFHE Location:** `D:\Manipal\Research\openfhe-install\`
  - Headers: `include\openfhe\`
  - Libraries: `lib\`
  
- **Your Project:** `D:\Manipal\Research\Implementation\`
  - Executable: `build\bin\Release\ckks_bootstrap_image.exe`

## Next Steps

Once everything builds successfully, the program will:
1. Generate a 64x64 dummy image
2. Encrypt it using CKKS
3. Perform homomorphic operations
4. Demonstrate bootstrapping
5. Save results as PGM image files
6. Show accuracy metrics

Estimated runtime: 30-60 seconds (bootstrapping key generation takes most of the time)
