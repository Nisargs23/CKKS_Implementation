#!/bin/bash

echo "================================================"
echo "  CKKS Bootstrapping Build Script (Linux/macOS)"
echo "================================================"
echo ""

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake is not installed"
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  macOS: brew install cmake"
    exit 1
fi

echo "CMake found: $(which cmake)"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo "Configuring project with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    echo "Please ensure OpenFHE is installed and accessible"
    cd ..
    exit 1
fi

# Build
echo ""
echo "Building project..."

# Detect number of cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    CORES=$(nproc)
fi

make -j$CORES

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
    cd ..
    exit 1
fi

cd ..

echo ""
echo "================================================"
echo "  Build completed successfully!"
echo "================================================"
echo ""
echo "To run the program:"
echo "  cd build/bin"
echo "  ./ckks_bootstrap_image"
echo ""
