# CKKS Bootstrapping for Image Processing & Encrypted Neural Net Inference

This project demonstrates **Fully Homomorphic Encryption (FHE)** using the **CKKS scheme** with **TRUE bootstrapping** from the OpenFHE library.

## Overview

Three progressively complex demonstrations:

| Demo | Image Size | Operations | Complexity |
|------|-----------|------------|------------|
| `ckks_bootstrap_image` | 16×16 (256 px) | Basic multiply + add | Proof of concept |
| `ckks_bootstrap_mnist` | 28×28 (784 px) | Basic multiply + add on real data | Real dataset |
| **`ckks_nn_highres`** | **64×64 (4096 px)** | **Neural network layer simulation** | **Research-grade** |

## Files

| File | Description |
|------|-------------|
| `main.cpp` | Basic demo with synthetic 16×16 image |
| `main_mnist.cpp` | MNIST handwritten digit processing |
| **`main_highres.cpp`** | **64×64 encrypted neural net inference** |
| `image_processor.h/cpp` | Image utility functions |
| `mnist_loader.h` | MNIST IDX format loader |
| `CMakeLists.txt` | CMake build configuration |

## Prerequisites

- **OpenFHE library** (built with FHE/bootstrapping support)
- **CMake** 3.15+
- **C++17** compiler
- **RAM**: 1-2 GB (basic/MNIST), **4+ GB** (high-res neural net)

## Building

```bash
cd final
mkdir build && cd build
cmake .. -DOPENFHE_INSTALL_DIR="D:/Manipal/Research/openfhe-install" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Repeatable RGB Benchmark (1/3/10 images + CSV)

From the repository root:

```powershell
.\final\benchmark_rgb.ps1
```

This script will:
- Configure/build `ckks_medmnist_rgb` with MSYS2 GCC in Release mode
- Set required runtime PATH entries (`openfhe-install\lib` and `C:\msys64\mingw64\bin`)
- Run image counts `1, 3, 10`
- Save raw logs and `summary.csv` under `final\benchmarks\<timestamp>`

Useful options:

```powershell
# Run only 1 and 3 images
.\final\benchmark_rgb.ps1 -ImageCounts 1,3

# Control outer channel execution strategy for bottleneck tuning
.\final\benchmark_rgb.ps1 -ChannelExecutionMode auto
.\final\benchmark_rgb.ps1 -ChannelExecutionMode parallel
.\final\benchmark_rgb.ps1 -ChannelExecutionMode sequential

# Optional: force bootstrap slot count (must be power-of-two and >= channel size)
.\final\benchmark_rgb.ps1 -BootstrapSlots 4096

# Optional: request a bootstrap correction factor (default/effective minimum is 10)
.\final\benchmark_rgb.ps1 -BootstrapCorrectionFactor 8

# Run cold and warm-start back-to-back
.\final\benchmark_rgb.ps1 -StartupMode both -ImageCounts 1,3,10

# Warm-start only (reuse serialized context/keys)
.\final\benchmark_rgb.ps1 -StartupMode warm -SkipBuild

# Clear cache before benchmarking
.\final\benchmark_rgb.ps1 -StartupMode both -ResetCache

# Reuse existing build artifacts
.\final\benchmark_rgb.ps1 -SkipBuild
```

`auto` now prefers outer channel parallelism when OpenMP has more than one
thread available. Use `sequential` only if your machine shows thread-contention
regressions.

By default, medmnist RGB/greyscale now chooses bootstrap slots as the smallest
power-of-two that fits a channel/image (`4096` for 64x64). You can override at
runtime with `CKKS_BOOTSTRAP_SLOTS`.

`CKKS_BOOTSTRAP_CORRECTION_FACTOR` controls bootstrap approximation depth.
For the current 5-layer x^2 pipeline, factors below `10` are rejected to avoid
runtime failures; use this mainly for reproducible experiments at `>= 10`.

Warm-start uses `CKKS_WARM_START=1` and `CKKS_CACHE_DIR` to load serialized
context + key material. The benchmark CSV reports both requested and effective
warm-start status (`WarmStartRequested`, `WarmStartUsed`).

The benchmark summary also includes `BootstrappingAvgPerChannelMs` so the
per-channel bootstrapping cost is easy to compare against the summed value.

## Running

### Basic Demo (16×16 synthetic image)
```bash
./bin/ckks_bootstrap_image
```

### MNIST Demo (28×28 handwritten digits)
```bash
./bin/ckks_bootstrap_mnist ../data/mnist/train-images-idx3-ubyte ../data/mnist/train-labels-idx1-ubyte 0
```

### High-Res Neural Net Demo (64×64)
```bash
# With synthetic test image
./bin/ckks_nn_highres

# With your own 64×64 PGM image
./bin/ckks_nn_highres path/to/your/image.pgm
```

---

## High-Res Neural Network Demo (NEW)

### What It Simulates

A **5-layer neural network** running entirely on **encrypted** data:

```
Input (64×64 image, 4096 pixels)
  │
  ├── Layer 1: weights × pixels + bias → x² activation      [2 levels]
  ├── Layer 2: weights × pixels + bias → x² activation      [2 levels]
  ├── Layer 3: weights × pixels + bias → x² activation      [2 levels]
  │
  ├── ★★★ BOOTSTRAPPING ★★★  (refresh ciphertext levels)
  │
  ├── Layer 4: weights × pixels + bias → x² activation      [2 levels]
  └── Layer 5: weights × pixels + bias → output             [1 level]

Output (encrypted inference result)
```

### Neural Network Operations

| Operation | FHE Implementation | Levels | Reference |
|-----------|-------------------|--------|-----------|
| **Linear transform** | `EvalMult(ciphertext, plaintext_weights)` | 1 | Standard FC layer |
| **Bias addition** | `EvalAdd(ciphertext, scalar)` | 0 | Standard bias |
| **Square activation** | `EvalMult(ciphertext, ciphertext)` | 1 | CryptoNets (ICML 2016) |

**Square activation (f(x) = x²)** is the standard FHE-friendly non-linearity from Microsoft Research's [CryptoNets paper](https://proceedings.mlr.press/v48/gilad-bachrach16.html). It's used because:
- It's a low-degree polynomial (degree 2)
- Consumes only 1 multiplicative level
- Works well as a non-linear activation in practice

### Cryptographic Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Ring Dimension | 16384 | Needed for 4096+ CKKS slots |
| Max Slots | 8192 | Fits 4096 pixels with headroom |
| Level Budget | {3, 3} | Bootstrap encoding/decoding levels |
| Scaling Mod | 50 bits | Precision per level |
| Security | Demo mode | Use `HEStd_128_classic` for production |
| Estimated RAM | ~4 GB | For bootstrapping key material |

### Expected Runtime

| Phase | Estimated Time |
|-------|---------------|
| Context + Key Setup | 10-60 seconds |
| Bootstrap Key Gen | 30-120 seconds |
| Encryption | < 1 second |
| 3 NN Layers (pre-bootstrap) | 1-5 seconds |
| **Bootstrapping** | **10-60 seconds** |
| 2 NN Layers (post-bootstrap) | 1-3 seconds |
| Decryption | < 1 second |
| **Total** | **~2-10 minutes** |

### Datasets for 64×64 Images

For real-world testing beyond the synthetic image:

| Dataset | Description | Download |
|---------|-------------|----------|
| **MedMNIST v2** | Medical images (pathology, dermatology, blood cells) at 64×64 | [medmnist.com](https://medmnist.com/) |
| **PathMNIST** | Colorectal cancer histology (9 tissue types) | Part of MedMNIST |
| **DermaMNIST** | Dermatoscopic images (7 skin conditions) | Part of MedMNIST |
| **BloodMNIST** | Blood cell microscopy images (8 cell types) | Part of MedMNIST |
| **Custom PGM** | Any 64×64 grayscale image in ASCII PGM format | Convert with ImageMagick: `convert input.png -resize 64x64 -colorspace Gray output.pgm` |

### Output Files

The high-res demo generates:
- `highres_original.pgm` - Input 64×64 image
- `highres_nn_output.pgm` - Raw encrypted inference output
- `highres_nn_normalized.pgm` - Normalized for visual comparison

---

## Key Concepts

### Why Bootstrapping is Essential for Neural Networks

| Without Bootstrapping | With Bootstrapping |
|----------------------|-------------------|
| Limited to ~5 multiplicative levels | **Unlimited** depth |
| Can run ~2-3 NN layers | Can run **any number** of layers |
| "Leveled" HE only | True FHE |
| Shallow models only | Deep networks possible |

### Level Consumption per NN Layer

```
Linear Layer (weights × x + bias):  1 level  (ciphertext-plaintext multiply + rescale)
Square Activation (x²):             1 level  (ciphertext-ciphertext multiply + rescale)
───────────────────────────────────────────
Total per layer with activation:     2 levels
```

With 10 available levels: max ~5 layers before bootstrapping is needed.

## Comparison Across Demos

| Metric | Basic (16×16) | MNIST (28×28) | High-Res NN (64×64) |
|--------|--------------|---------------|---------------------|
| Pixels | 256 | 784 | 4096 |
| Ring Dim | 4096 | 4096 | **16384** |
| Slots Used | 512 | 1024 | 4096 |
| Operations | Multiply + Add | Multiply + Add | **NN layers + activations** |
| Op Type | Scalar | Scalar | **Vector (per-pixel weights)** |
| Bootstrap Time | ~100-500 ms | ~100-500 ms | **~10-60 seconds** |
| Total Time | ~10-30 sec | ~10-30 sec | **~2-10 min** |
| Use Case | Proof of concept | Real data validation | **Encrypted ML inference** |

## References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [CKKS Scheme](https://eprint.iacr.org/2016/421) - Cheon, Kim, Kim, Song (2016)
- [Bootstrapping for CKKS](https://eprint.iacr.org/2018/153) - Cheon et al. (2018)
- [CryptoNets](https://proceedings.mlr.press/v48/gilad-bachrach16.html) - Gilad-Bachrach et al., ICML 2016 (square activation)
- [MedMNIST v2](https://medmnist.com/) - Medical image datasets at 64×64
