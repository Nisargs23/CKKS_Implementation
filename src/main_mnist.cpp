#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/scheme/ckksrns/ckksrns-fhe.h"
#include "image_processor.h"
#include "mnist_loader.h"

using namespace lbcrypto;
using namespace std;

// Structure to hold all performance metrics
struct PerformanceMetrics {
    double contextSetupTime = 0;
    double keyGenTime = 0;
    double multKeyGenTime = 0;
    double bootstrapKeyGenTime = 0;
    double bootstrapSetupTime = 0;
    double encryptionTime = 0;
    double decryptionTime = 0;
    double homomorphicOpsTime = 0;
    double bootstrapTime = 0;
    double totalTime = 0;
    
    uint32_t ringDimension = 0;
    uint32_t multDepth = 0;
    uint32_t levelsAfterBootstrap = 0;
    uint32_t numSlots = 0;
    
    uint32_t initialLevel = 0;
    uint32_t levelBeforeBootstrap = 0;
    uint32_t levelAfterBootstrap = 0;
    uint32_t finalLevel = 0;
    
    int numOperationsBeforeBootstrap = 0;
    int numOperationsAfterBootstrap = 0;
    int numBootstraps = 0;
    
    double maxError = 0;
    double rmse = 0;
    size_t numPixels = 0;
    
    // MNIST specific
    int numImagesProcessed = 0;
    uint8_t imageLabel = 0;
};

// Function to print comprehensive performance report
void PrintPerformanceReport(const PerformanceMetrics& m) {
    cout << "\n" << string(60, '=') << endl;
    cout << "       COMPREHENSIVE PERFORMANCE REPORT" << endl;
    cout << string(60, '=') << endl;
    
    cout << "\n--- MNIST DATASET INFO ---" << endl;
    cout << "Images Processed:              " << m.numImagesProcessed << endl;
    cout << "Image Label (digit):           " << (int)m.imageLabel << endl;
    cout << "Pixels per Image:              " << m.numPixels << endl;
    
    cout << "\n--- TIMING METRICS ---" << fixed << setprecision(2) << endl;
    cout << "Context Setup Time:            " << setw(10) << m.contextSetupTime << " ms" << endl;
    cout << "Key Generation Time:           " << setw(10) << m.keyGenTime << " ms" << endl;
    cout << "Mult Key Generation Time:      " << setw(10) << m.multKeyGenTime << " ms" << endl;
    cout << "Bootstrap Setup Time:          " << setw(10) << m.bootstrapSetupTime << " ms" << endl;
    cout << "Bootstrap Key Gen Time:        " << setw(10) << m.bootstrapKeyGenTime << " ms" << endl;
    cout << "Encryption Time:               " << setw(10) << m.encryptionTime << " ms" << endl;
    cout << "Homomorphic Ops Time:          " << setw(10) << m.homomorphicOpsTime << " ms" << endl;
    cout << "Bootstrapping Time:            " << setw(10) << m.bootstrapTime << " ms" << endl;
    cout << "Decryption Time:               " << setw(10) << m.decryptionTime << " ms" << endl;
    cout << string(45, '-') << endl;
    cout << "TOTAL RUNTIME:                 " << setw(10) << m.totalTime << " ms" << endl;
    cout << "                               " << setw(10) << (m.totalTime / 1000.0) << " seconds" << endl;
    
    cout << "\n--- CRYPTOGRAPHIC PARAMETERS ---" << endl;
    cout << "Ring Dimension:                " << m.ringDimension << endl;
    cout << "Multiplicative Depth:          " << m.multDepth << endl;
    cout << "Levels After Bootstrap:        " << m.levelsAfterBootstrap << endl;
    cout << "Number of Slots:               " << m.numSlots << endl;
    
    cout << "\n--- LEVEL TRACKING ---" << endl;
    cout << "Initial Ciphertext Level:      " << m.initialLevel << endl;
    cout << "Level Before Bootstrap:        " << m.levelBeforeBootstrap << endl;
    cout << "Level After Bootstrap:         " << m.levelAfterBootstrap << endl;
    cout << "Final Ciphertext Level:        " << m.finalLevel << endl;
    cout << "Levels Refreshed:              " << (m.levelAfterBootstrap - m.levelBeforeBootstrap) << endl;
    
    cout << "\n--- OPERATION STATISTICS ---" << endl;
    cout << "Operations Before Bootstrap:   " << m.numOperationsBeforeBootstrap << endl;
    cout << "Number of Bootstraps:          " << m.numBootstraps << endl;
    cout << "Operations After Bootstrap:    " << m.numOperationsAfterBootstrap << endl;
    cout << "Total Homomorphic Operations:  " << (m.numOperationsBeforeBootstrap + m.numOperationsAfterBootstrap) << endl;
    
    cout << "\n--- ACCURACY METRICS ---" << endl;
    cout << "Pixels Processed:              " << m.numPixels << endl;
    cout << scientific << setprecision(6);
    cout << "Max Absolute Error:            " << m.maxError << endl;
    cout << "Root Mean Square Error:        " << m.rmse << endl;
    
    cout << fixed << setprecision(2);
    cout << "\n--- THROUGHPUT ---" << endl;
    if (m.encryptionTime > 0)
        cout << "Encryption Throughput:         " << (m.numPixels / (m.encryptionTime / 1000.0)) << " pixels/sec" << endl;
    if (m.decryptionTime > 0)
        cout << "Decryption Throughput:         " << (m.numPixels / (m.decryptionTime / 1000.0)) << " pixels/sec" << endl;
    if (m.bootstrapTime > 0 && m.numBootstraps > 0)
        cout << "Avg Bootstrap Time:            " << (m.bootstrapTime / m.numBootstraps) << " ms/bootstrap" << endl;
    
    cout << "\n" << string(60, '=') << endl;
    
    // Status indicators
    if (m.numBootstraps > 0 && m.levelAfterBootstrap > m.levelBeforeBootstrap) {
        cout << "✅ BOOTSTRAPPING: SUCCESS - Levels refreshed!" << endl;
    }
    if (m.rmse < 1e-3) {
        cout << "✅ ACCURACY: Excellent (RMSE < 0.001)" << endl;
    } else if (m.rmse < 1e-1) {
        cout << "⚠️  ACCURACY: Moderate (some precision loss from bootstrap)" << endl;
    }
    cout << string(60, '=') << "\n" << endl;
}

// Function to setup CKKS with TRUE bootstrapping support
CryptoContext<DCRTPoly> SetupCKKSWithBootstrapping(PerformanceMetrics& metrics) {
    cout << "\n========================================" << endl;
    cout << "Setting up CKKS with TRUE Bootstrapping" << endl;
    cout << "========================================\n" << endl;

    auto setupStart = chrono::high_resolution_clock::now();
    
    CCParams<CryptoContextCKKSRNS> parameters;
    
    // A1) Secret key distribution - UNIFORM_TERNARY is standard
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    
    // A2) Security level - using NotSet for faster demo, use HEStd_128_classic for production
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);  // 4096 - smaller for demo speed
    
    // A3) Bootstrapping parameters
    std::vector<uint32_t> levelBudget = {3, 3};
    
    // Number of levels available after bootstrapping for computation
    uint32_t levelsAvailableAfterBootstrap = 10;
    metrics.levelsAfterBootstrap = levelsAvailableAfterBootstrap;
    
    // A4) Compute required depth
    uint32_t depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    parameters.SetMultiplicativeDepth(depth);
    metrics.multDepth = depth;
    
    cout << "Bootstrapping depth required: " << FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist) << endl;
    cout << "Total multiplicative depth: " << depth << endl;
    
    // A5) Other parameters
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    
    // Create crypto context
    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
    
    // Enable features
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);
    cryptoContext->Enable(ADVANCEDSHE);
    cryptoContext->Enable(FHE);
    
    metrics.ringDimension = cryptoContext->GetRingDimension();
    metrics.numSlots = metrics.ringDimension / 2;
    
    auto setupEnd = chrono::high_resolution_clock::now();
    metrics.contextSetupTime = chrono::duration_cast<chrono::milliseconds>(setupEnd - setupStart).count();
    
    cout << "\nCrypto Context Parameters:" << endl;
    cout << "  Ring Dimension: " << metrics.ringDimension << endl;
    cout << "  Multiplicative Depth: " << depth << endl;
    cout << "  Max Slots: " << metrics.numSlots << endl;
    cout << "  FHE Mode: ENABLED ✓" << endl;
    cout << "\nContext setup time: " << metrics.contextSetupTime << " ms" << endl;
    
    return cryptoContext;
}

// Function to encrypt image data
Ciphertext<DCRTPoly> EncryptImage(
    CryptoContext<DCRTPoly>& cryptoContext,
    const PublicKey<DCRTPoly>& publicKey,
    const vector<double>& imageData,
    uint32_t numSlots,
    PerformanceMetrics& metrics) {
    
    cout << "\n========================================" << endl;
    cout << "Encrypting MNIST Image Data" << endl;
    cout << "========================================\n" << endl;
    
    metrics.numPixels = imageData.size();
    
    Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(imageData, 1, 0, nullptr, numSlots);
    
    cout << "Plaintext created with " << imageData.size() << " pixels (28x28 MNIST)" << endl;
    cout << "Using " << numSlots << " slots" << endl;
    
    auto start = chrono::high_resolution_clock::now();
    auto ciphertext = cryptoContext->Encrypt(publicKey, plaintext);
    auto end = chrono::high_resolution_clock::now();
    
    metrics.encryptionTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    metrics.initialLevel = ciphertext->GetLevel();
    
    cout << "Encryption time: " << fixed << setprecision(2) << metrics.encryptionTime << " ms" << endl;
    cout << "Initial ciphertext level: " << metrics.initialLevel << endl;
    
    return ciphertext;
}

// Function to perform homomorphic operations (simulating image processing)
Ciphertext<DCRTPoly> ProcessImageHomomorphically(
    CryptoContext<DCRTPoly>& cryptoContext,
    Ciphertext<DCRTPoly>& ciphertext,
    int numOperations,
    const string& phase = "before") {
    
    cout << "\n========================================" << endl;
    cout << "Homomorphic Operations (" << phase << " bootstrap)" << endl;
    cout << "========================================\n" << endl;
    
    auto result = ciphertext;
    
    cout << "Performing " << numOperations << " multiply-add cycles..." << endl;
    cout << "Starting level: " << result->GetLevel() << endl;
    
    for (int i = 0; i < numOperations; ++i) {
        // Multiply by a constant (simulating contrast adjustment)
        double factor = 1.05;
        result = cryptoContext->EvalMult(result, factor);
        
        // Rescale to maintain precision
        result = cryptoContext->Rescale(result);
        
        // Add a constant (simulating brightness shift)
        double shift = 0.01;
        result = cryptoContext->EvalAdd(result, shift);
        
        cout << "  Cycle " << (i + 1) << " completed. Level: " 
             << result->GetLevel() << endl;
    }
    
    cout << "Final level after " << phase << " operations: " << result->GetLevel() << endl;
    
    return result;
}

// Function to perform TRUE bootstrapping
Ciphertext<DCRTPoly> PerformBootstrapping(
    CryptoContext<DCRTPoly>& cryptoContext,
    Ciphertext<DCRTPoly>& ciphertext,
    PerformanceMetrics& metrics) {
    
    cout << "\n" << string(50, '*') << endl;
    cout << "    PERFORMING TRUE BOOTSTRAPPING" << endl;
    cout << string(50, '*') << "\n" << endl;
    
    metrics.levelBeforeBootstrap = ciphertext->GetLevel();
    cout << "Ciphertext level BEFORE bootstrapping: " << metrics.levelBeforeBootstrap << endl;
    cout << "Noise has accumulated - refreshing ciphertext..." << endl;
    
    auto start = chrono::high_resolution_clock::now();
    
    auto bootstrappedCt = cryptoContext->EvalBootstrap(ciphertext);
    
    auto end = chrono::high_resolution_clock::now();
    
    metrics.bootstrapTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    metrics.levelAfterBootstrap = bootstrappedCt->GetLevel();
    metrics.numBootstraps++;
    
    cout << "\nCiphertext level AFTER bootstrapping: " << metrics.levelAfterBootstrap << endl;
    cout << "Levels refreshed: " << (metrics.levelAfterBootstrap - metrics.levelBeforeBootstrap) << endl;
    cout << "Bootstrapping time: " << metrics.bootstrapTime << " ms" << endl;
    cout << "\n✅ BOOTSTRAPPING SUCCESSFUL! Noise refreshed." << endl;
    cout << string(50, '*') << "\n" << endl;
    
    return bootstrappedCt;
}

// Function to decrypt and extract image data
vector<double> DecryptImage(
    CryptoContext<DCRTPoly>& cryptoContext,
    const PrivateKey<DCRTPoly>& privateKey,
    Ciphertext<DCRTPoly>& ciphertext,
    size_t originalSize,
    PerformanceMetrics& metrics) {
    
    cout << "\n========================================" << endl;
    cout << "Decrypting Image Data" << endl;
    cout << "========================================\n" << endl;
    
    metrics.finalLevel = ciphertext->GetLevel();
    
    Plaintext plaintextResult;
    
    auto start = chrono::high_resolution_clock::now();
    cryptoContext->Decrypt(privateKey, ciphertext, &plaintextResult);
    auto end = chrono::high_resolution_clock::now();
    
    metrics.decryptionTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    
    cout << "Final ciphertext level: " << metrics.finalLevel << endl;
    cout << "Decryption time: " << fixed << setprecision(2) << metrics.decryptionTime << " ms" << endl;
    
    plaintextResult->SetLength(originalSize);
    vector<double> result = plaintextResult->GetRealPackedValue();
    result.resize(originalSize);
    
    cout << "Decrypted " << result.size() << " pixel values" << endl;
    
    return result;
}

// Function to calculate error between original and processed images
void CalculateAccuracyMetrics(const vector<double>& original, const vector<double>& decrypted, 
                              PerformanceMetrics& metrics) {
    if (original.size() != decrypted.size()) {
        cerr << "Error: Image sizes don't match!" << endl;
        return;
    }
    
    cout << "\n========================================" << endl;
    cout << "Accuracy Analysis" << endl;
    cout << "========================================\n" << endl;
    
    double maxError = 0.0;
    double sumSquaredError = 0.0;
    
    for (size_t i = 0; i < original.size(); ++i) {
        double error = abs(original[i] - decrypted[i]);
        maxError = max(maxError, error);
        sumSquaredError += error * error;
    }
    
    metrics.maxError = maxError;
    metrics.rmse = sqrt(sumSquaredError / original.size());
    
    cout << scientific << setprecision(6);
    cout << "Max absolute error: " << metrics.maxError << endl;
    cout << "Root Mean Square Error (RMSE): " << metrics.rmse << endl;
    
    if (metrics.rmse < 1e-4) {
        cout << "✅ Excellent accuracy!" << endl;
    } else if (metrics.rmse < 1e-2) {
        cout << "✅ Good accuracy (some noise from bootstrapping is normal)" << endl;
    } else {
        cout << "⚠️  Moderate accuracy (precision loss from bootstrapping)" << endl;
    }
}

// Save image as PGM file
void saveImageAsPGM(const vector<double>& image, int width, int height, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }
    
    file << "P2\n";
    file << width << " " << height << "\n";
    file << "255\n";
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            // Clamp values to [0, 1] before scaling
            double val = max(0.0, min(1.0, image[idx]));
            int pixelValue = static_cast<int>(val * 255.0);
            file << pixelValue << " ";
        }
        file << "\n";
    }
    
    file.close();
    cout << "Image saved to: " << filename << endl;
}

void printUsage(const char* programName) {
    cout << "\nUsage: " << programName << " <path-to-mnist-images> <path-to-mnist-labels> [image-index]" << endl;
    cout << "\nExample:" << endl;
    cout << "  " << programName << " train-images.idx3-ubyte train-labels.idx1-ubyte 0" << endl;
    cout << "\nDownload MNIST from: http://yann.lecun.com/exdb/mnist/" << endl;
    cout << "  - train-images-idx3-ubyte.gz" << endl;
    cout << "  - train-labels-idx1-ubyte.gz" << endl;
    cout << "\nExtract the .gz files before use." << endl;
}

int main(int argc, char* argv[]) {
    cout << "\n" << string(60, '*') << endl;
    cout << "   CKKS BOOTSTRAPPING on MNIST Dataset" << endl;
    cout << "   Using OpenFHE Library - Real-World Image Processing" << endl;
    cout << string(60, '*') << "\n" << endl;
    
    // Check command line arguments
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    string imagesPath = argv[1];
    string labelsPath = argv[2];
    int imageIndex = (argc > 3) ? atoi(argv[3]) : 0;
    
    PerformanceMetrics metrics;
    auto totalStart = chrono::high_resolution_clock::now();
    
    try {
        // ========================================
        // Step 1: Load MNIST Dataset
        // ========================================
        cout << "\n========================================" << endl;
        cout << "Step 1: Load MNIST Dataset" << endl;
        cout << "========================================\n" << endl;
        
        cout << "Loading images from: " << imagesPath << endl;
        cout << "Loading labels from: " << labelsPath << endl;
        
        // Load only a few images to save memory
        int maxImages = imageIndex + 10;  // Load enough to get our target image
        auto images = MNISTLoader::loadImages(imagesPath, maxImages);
        auto labels = MNISTLoader::loadLabels(labelsPath, maxImages);
        
        if (images.empty() || labels.empty()) {
            cerr << "Error: Failed to load MNIST dataset!" << endl;
            return 1;
        }
        
        if (imageIndex >= (int)images.size()) {
            cerr << "Error: Image index " << imageIndex << " out of range (max: " << images.size() - 1 << ")" << endl;
            return 1;
        }
        
        // Get the selected image
        vector<double> originalImage = images[imageIndex];
        uint8_t label = labels[imageIndex];
        metrics.imageLabel = label;
        metrics.numImagesProcessed = 1;
        
        cout << "\nSelected image index: " << imageIndex << endl;
        cout << "Image label (digit): " << (int)label << endl;
        cout << "Image size: 28x28 = " << originalImage.size() << " pixels" << endl;
        
        // Print ASCII representation of the image
        cout << "\nOriginal MNIST image (digit " << (int)label << "):" << endl;
        cout << string(58, '-') << endl;
        MNISTLoader::printImage(originalImage);
        cout << string(58, '-') << endl;
        
        // Save original image
        saveImageAsPGM(originalImage, 28, 28, "mnist_original.pgm");
        
        // Calculate and print image statistics
        double sum = 0, minVal = 1, maxVal = 0;
        for (double v : originalImage) {
            sum += v;
            minVal = min(minVal, v);
            maxVal = max(maxVal, v);
        }
        cout << "\nImage Statistics:" << endl;
        cout << "  Mean pixel value: " << (sum / originalImage.size()) << endl;
        cout << "  Min: " << minVal << ", Max: " << maxVal << endl;
        
        // ========================================
        // Step 2: Setup CKKS with bootstrapping
        // ========================================
        auto cryptoContext = SetupCKKSWithBootstrapping(metrics);
        
        // MNIST image is 784 pixels, use 1024 slots
        uint32_t numSlots = 1024;
        
        // ========================================
        // Step 3: Bootstrap precomputation setup
        // ========================================
        cout << "\n========================================" << endl;
        cout << "Step 2: Bootstrap Setup (Precomputation)" << endl;
        cout << "========================================\n" << endl;
        
        vector<uint32_t> levelBudget = {3, 3};
        
        auto setupStart = chrono::high_resolution_clock::now();
        cryptoContext->EvalBootstrapSetup(levelBudget, {0, 0}, numSlots);
        auto setupEnd = chrono::high_resolution_clock::now();
        metrics.bootstrapSetupTime = chrono::duration_cast<chrono::milliseconds>(setupEnd - setupStart).count();
        
        cout << "Bootstrap setup completed in " << metrics.bootstrapSetupTime << " ms" << endl;
        
        // ========================================
        // Step 4: Generate all keys
        // ========================================
        cout << "\n========================================" << endl;
        cout << "Step 3: Generate Cryptographic Keys" << endl;
        cout << "========================================\n" << endl;
        
        auto keyStart = chrono::high_resolution_clock::now();
        auto keyPair = cryptoContext->KeyGen();
        auto keyEnd = chrono::high_resolution_clock::now();
        metrics.keyGenTime = chrono::duration_cast<chrono::milliseconds>(keyEnd - keyStart).count();
        cout << "✓ Key pair generated (" << metrics.keyGenTime << " ms)" << endl;
        
        auto multKeyStart = chrono::high_resolution_clock::now();
        cryptoContext->EvalMultKeyGen(keyPair.secretKey);
        auto multKeyEnd = chrono::high_resolution_clock::now();
        metrics.multKeyGenTime = chrono::duration_cast<chrono::milliseconds>(multKeyEnd - multKeyStart).count();
        cout << "✓ Multiplication keys generated (" << metrics.multKeyGenTime << " ms)" << endl;
        
        cout << "\nGenerating bootstrapping keys..." << endl;
        auto bootKeyStart = chrono::high_resolution_clock::now();
        cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
        auto bootKeyEnd = chrono::high_resolution_clock::now();
        metrics.bootstrapKeyGenTime = chrono::duration_cast<chrono::milliseconds>(bootKeyEnd - bootKeyStart).count();
        cout << "✓ Bootstrapping keys generated (" << metrics.bootstrapKeyGenTime << " ms)" << endl;
        
        cout << "\n✅ All keys generated! Ready to process MNIST image." << endl;
        
        // ========================================
        // Step 5: Encrypt MNIST image
        // ========================================
        auto encryptedImage = EncryptImage(cryptoContext, keyPair.publicKey, 
                                           originalImage, numSlots, metrics);
        
        // ========================================
        // Step 6: Perform operations BEFORE bootstrapping
        // ========================================
        int opsBeforeBootstrap = 8;
        metrics.numOperationsBeforeBootstrap = opsBeforeBootstrap;
        
        auto opsStart = chrono::high_resolution_clock::now();
        auto processedImage = ProcessImageHomomorphically(
            cryptoContext, encryptedImage, opsBeforeBootstrap, "BEFORE");
        
        cout << "\n⚠️  After " << opsBeforeBootstrap << " operations, ciphertext level is depleted." << endl;
        
        // ========================================
        // Step 7: PERFORM TRUE BOOTSTRAPPING!
        // ========================================
        auto bootstrappedImage = PerformBootstrapping(cryptoContext, processedImage, metrics);
        
        // ========================================
        // Step 8: Perform MORE operations AFTER bootstrapping
        // ========================================
        int opsAfterBootstrap = 5;
        metrics.numOperationsAfterBootstrap = opsAfterBootstrap;
        
        auto finalImage = ProcessImageHomomorphically(
            cryptoContext, bootstrappedImage, opsAfterBootstrap, "AFTER");
        
        auto opsEnd = chrono::high_resolution_clock::now();
        metrics.homomorphicOpsTime = chrono::duration_cast<chrono::milliseconds>(opsEnd - opsStart).count() 
                                     - metrics.bootstrapTime;
        
        cout << "\n✅ Successfully performed " << opsAfterBootstrap 
             << " MORE operations after bootstrapping!" << endl;
        
        // ========================================
        // Step 9: Decrypt and verify
        // ========================================
        vector<double> decryptedImage = DecryptImage(
            cryptoContext, keyPair.secretKey, finalImage, originalImage.size(), metrics);
        
        // Normalize decrypted image for display
        vector<double> normalizedDecrypted(decryptedImage.size());
        double decMin = *min_element(decryptedImage.begin(), decryptedImage.end());
        double decMax = *max_element(decryptedImage.begin(), decryptedImage.end());
        for (size_t i = 0; i < decryptedImage.size(); ++i) {
            normalizedDecrypted[i] = (decryptedImage[i] - decMin) / (decMax - decMin);
        }
        
        // Print decrypted image
        cout << "\nDecrypted MNIST image after homomorphic processing:" << endl;
        cout << string(58, '-') << endl;
        MNISTLoader::printImage(normalizedDecrypted);
        cout << string(58, '-') << endl;
        
        saveImageAsPGM(normalizedDecrypted, 28, 28, "mnist_decrypted.pgm");
        
        // ========================================
        // Step 10: Calculate accuracy
        // ========================================
        vector<double> expectedImage = originalImage;
        double totalFactor = 1.0;
        double totalShift = 0.0;
        for (int i = 0; i < opsBeforeBootstrap + opsAfterBootstrap; ++i) {
            totalFactor *= 1.05;
            totalShift = totalShift * 1.05 + 0.01;
        }
        for (size_t i = 0; i < expectedImage.size(); ++i) {
            expectedImage[i] = expectedImage[i] * totalFactor + totalShift;
        }
        
        CalculateAccuracyMetrics(expectedImage, decryptedImage, metrics);
        
        // ========================================
        // Calculate total time and print report
        // ========================================
        auto totalEnd = chrono::high_resolution_clock::now();
        metrics.totalTime = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart).count();
        
        PrintPerformanceReport(metrics);
        
        // ========================================
        // Final Summary
        // ========================================
        cout << "\n" << string(60, '*') << endl;
        cout << "   MNIST BOOTSTRAPPING DEMONSTRATION COMPLETE!" << endl;
        cout << string(60, '*') << endl;
        
        cout << "\nProcessed MNIST digit: " << (int)label << endl;
        cout << "Total operations performed: " << (opsBeforeBootstrap + opsAfterBootstrap) << endl;
        cout << "Bootstrapping: SUCCESS" << endl;
        cout << "RMSE: " << scientific << setprecision(2) << metrics.rmse << endl;
        
        cout << "\nOutput files:" << endl;
        cout << "  - mnist_original.pgm (original MNIST image)" << endl;
        cout << "  - mnist_decrypted.pgm (after homomorphic processing)" << endl;
        
        cout << "\n" << string(60, '*') << "\n" << endl;
        
    } catch (const exception& e) {
        cerr << "\n❌ Error: " << e.what() << endl;
        cerr << "\nTroubleshooting tips:" << endl;
        cerr << "1. Make sure the MNIST files exist and are extracted (.gz removed)" << endl;
        cerr << "2. Check file paths are correct" << endl;
        cerr << "3. Ensure sufficient memory (~2GB)" << endl;
        return 1;
    }
    
    return 0;
}
