# CKKS Bootstrapping for Image Processing

This project demonstrates **bootstrapping** in Fully Homomorphic Encryption (FHE) using the **CKKS scheme** with **OpenFHE library**. It processes images while encrypted, showcasing unlimited computational depth through bootstrapping.

## What is Bootstrapping?

Bootstrapping is a technique in FHE that "refreshes" the noise in a ciphertext, allowing unlimited homomorphic operations. Without bootstrapping, you're limited by the multiplicative depth of the scheme. With bootstrapping, you can:
- Perform unlimited multiplications and additions on encrypted data
- Build complex computational circuits
- Process data without ever decrypting it

## Features

- ✅ CKKS scheme implementation with OpenFHE
- ✅ Full bootstrapping support
- ✅ Image encryption and processing
- ✅ Homomorphic operations on encrypted images
- ✅ Noise refresh through bootstrapping
- ✅ Accuracy analysis and error metrics
- ✅ Dummy image generation for testing

## Prerequisites

### 1. Install OpenFHE

#### On Windows (using MSYS2 - recommended):
```powershell
# Step 1: Install MSYS2 and build tools
winget install --id MSYS2.MSYS2 -e

# Open a NEW PowerShell window and run:
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -Sy --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja"

# Step 2: Clone OpenFHE (if not already done)
cd D:\Manipal\Research  # or your preferred directory
git clone https://github.com/openfheorg/openfhe-development.git

# Step 3: Build OpenFHE using the automated script
cd Implementation
.\build-openfhe.ps1
```

**Note:** The build takes 20-30 minutes. The script handles everything automatically.

#### On Linux (Ubuntu/Debian):
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential cmake git

# Clone and build OpenFHE
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

#### On macOS:
```bash
# Install dependencies
brew install cmake

# Clone and build OpenFHE
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### 2. Required Tools
- CMake (>= 3.15)
- C++17 compatible compiler
  - Windows: Visual Studio 2019 or later
  - Linux: GCC 9+ or Clang 10+
  - macOS: Xcode Command Line Tools

## Building the Project

### On Windows (MSYS2):
```powershell
# After OpenFHE is built, build your project:
cd D:\Manipal\Research\Implementation
.\build-with-msys2.ps1

# Run the program:
cd build\bin
$env:Path = "C:\msys64\mingw64\bin;$env:Path"
.\ckks_bootstrap_image.exe
```

### On Linux/macOS:
```bash
# Navigate to project directory
cd /path/to/Implementation

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)  # Linux
# or
make -j$(sysctl -n hw.ncpu)  # macOS

# Run
./bin/ckks_bootstrap_image
```

## Project Structure

```
Implementation/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── src/
│   ├── main.cpp            # Main program demonstrating bootstrapping
│   ├── image_processor.h   # Image processing header
│   └── image_processor.cpp # Image processing implementation
└── build/                  # Build directory (created during compilation)
```

## How It Works

### 1. **Image Generation**
Creates a 64x64 pixel dummy image with gradients and geometric shapes.

### 2. **CKKS Setup with Bootstrapping**
Configures the CKKS cryptosystem with:
- Multiplicative depth: 10
- Security level: 128-bit classical security
- Batch size: 8192 slots
- Bootstrapping enabled with THIN_BOOT mode

### 3. **Key Generation**
Generates:
- Public/private key pair
- Multiplication (relinearization) keys
- **Bootstrapping keys** (enables noise refresh)

### 4. **Encryption**
Encrypts the image pixels into a single CKKS ciphertext.

### 5. **Homomorphic Operations**
Performs multiple operations on encrypted data:
- Multiplication (brightness adjustment)
- Addition (brightness shift)
- These operations consume the multiplicative depth

### 6. **Bootstrapping**
When the noise level gets too high:
- Applies the bootstrapping operation
- Refreshes the ciphertext
- Resets to high level allowing more operations

### 7. **Additional Operations**
Demonstrates that after bootstrapping, you can continue computing on the refreshed ciphertext.

### 8. **Decryption & Verification**
Decrypts the final result and compares with the expected output to measure accuracy.

## Output

The program generates:
- `original_image.pgm` - The original dummy image
- `decrypted_image.pgm` - The decrypted image after homomorphic operations and bootstrapping
- Console output showing:
  - Image statistics
  - Encryption/decryption times
  - Bootstrapping time and statistics
  - Accuracy metrics (RMSE, max error)

### Sample Output:
```
************************************************
   CKKS Bootstrapping for Image Processing
   Using OpenFHE Library
************************************************

========================================
Step 1: Generate Dummy Image
========================================

Original Image Statistics:
  Size: 64x64 (4096 pixels)
  Mean: 0.512
  ...

========================================
Setting up CKKS with Bootstrapping
========================================

Ring Dimension: 32768
Multiplicative Depth: 10
Batch Size: 8192
Bootstrapping enabled successfully!

...

========================================
Performing Bootstrapping
========================================

Ciphertext level before bootstrapping: 2
Ciphertext level after bootstrapping: 9
Bootstrapping time: 1523 ms
Bootstrapping successful! Noise refreshed.

...

************************************************
   Bootstrapping Demonstration Complete!
************************************************

Summary:
- Image size: 64x64
- Operations before bootstrapping: 3 cycles
- Bootstrapping: SUCCESS
- Operations after bootstrapping: 2 additional operations
- Final RMSE: 0.000123

Bootstrapping allows unlimited depth operations!
```

## Key Concepts Demonstrated

### 1. **Multiplicative Depth**
Each homomorphic multiplication consumes one level of depth. CKKS without bootstrapping is limited to a fixed depth (e.g., 10 multiplications).

### 2. **Noise Growth**
Homomorphic operations add noise to ciphertexts. Too much noise prevents correct decryption.

### 3. **Bootstrapping Operation**
- Takes a "noisy" ciphertext at low level
- Homomorphically evaluates the decryption circuit
- Outputs a "fresh" ciphertext at high level
- Allows unlimited operations by repeating this process

### 4. **Trade-offs**
- **Pros**: Unlimited computational depth
- **Cons**: Bootstrapping is computationally expensive (1-2 seconds per operation)

## Performance Notes

- **Bootstrapping key generation**: 10-30 seconds (one-time cost)
- **Encryption**: <100 ms per image
- **Homomorphic operations**: <10 ms per operation
- **Bootstrapping**: 1-3 seconds per refresh
- **Decryption**: <100 ms per image

Performance varies based on:
- Hardware (CPU speed, cores)
- Parameter selection (security level, ring dimension)
- Bootstrapping configuration

## Customization

### Change Image Size
In `main.cpp`, modify:
```cpp
int imageWidth = 128;   // Change from 64
int imageHeight = 128;  // Change from 64
```

### Adjust Security Parameters
In `main.cpp`, modify the `SetupCKKSBootstrapping()` function:
```cpp
uint32_t multDepth = 15;        // More depth
uint32_t scaleModSize = 60;     // Larger scale
parameters.SetSecurityLevel(HEStd_256_classic);  // Higher security
```

### More Operations
Change the number of operation cycles:
```cpp
int numOperations = 5;  // Increase from 3
```

## Troubleshooting

### OpenFHE Not Found
```
CMake Error: Could not find a package configuration file provided by "OpenFHE"
```
**Solution**: Ensure OpenFHE is installed and CMake can find it:
- Set `CMAKE_PREFIX_PATH` to OpenFHE installation directory
- Or use vcpkg toolchain file on Windows

### Out of Memory
If bootstrapping fails with memory errors, reduce batch size:
```cpp
uint32_t batchSize = 4096;  // Reduce from 8192
```

### Compilation Errors
Ensure you're using C++17 or later:
```bash
cmake .. -DCMAKE_CXX_STANDARD=17
```

## References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [CKKS Paper](https://eprint.iacr.org/2016/421.pdf)
- [Bootstrapping in CKKS](https://eprint.iacr.org/2018/153.pdf)

## License

This is a research implementation for educational purposes.

## Author

Research Implementation - Manipal
Date: January 2026
