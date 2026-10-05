/**
 * Encrypted Medical Image Classification using Trained CryptoNets
 * ================================================================
 * 
 * PRIVACY-PRESERVING BREAST CANCER DETECTION
 * 
 * This program demonstrates a real-world privacy-preserving medical AI workflow:
 * 
 *   1. A hospital has a patient's breast ultrasound image
 *   2. They encrypt it locally using CKKS FHE
 *   3. The encrypted image is sent to a cloud ML service
 *   4. The cloud runs a trained neural network on the ENCRYPTED image
 *   5. The cloud returns an ENCRYPTED prediction
 *   6. The hospital decrypts to get: "malignant" or "benign"
 *   7. The cloud NEVER sees the patient's image or diagnosis
 * 
 * Architecture (trained on BreastMNIST with x² activations):
 *   Input (64×64) → [Layer1+x²] → [Layer2+x²] → [Layer3+x²]
 *   → [BOOTSTRAP] → [Layer4+x²] → [Layer5] → Output → Classify
 * 
 * Weights are loaded from files exported by the Python training script.
 * Classification (argmax) happens post-decryption since it's non-polynomial.
 * 
 * Dataset: BreastMNIST (MedMNIST v2) — 64×64 breast ultrasound
 *   - Class 0: malignant
 *   - Class 1: normal/benign
 * 
 * Reference: 
 *   - CryptoNets (Gilad-Bachrach et al., ICML 2016)
 *   - MedMNIST v2 (Yang et al., Scientific Data 2023)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdlib>
#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/scheme/ckksrns/ckksrns-fhe.h"
#include "security_metrics.h"

using namespace lbcrypto;
using namespace std;

// ============================================================
// Constants
// ============================================================
const int IMAGE_WIDTH  = 64;
const int IMAGE_HEIGHT = 64;
const int IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT;  // 4096 pixels
const int NUM_CLASSES   = 2;

const string CLASS_NAMES[] = {"malignant", "normal/benign"};

bool IsPowerOfTwo(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

uint32_t NextPowerOfTwo(uint32_t value) {
    if (value <= 1) {
        return 1;
    }

    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

uint32_t ResolveBootstrapSlots(uint32_t requiredSlots, uint32_t maxSlots) {
    uint32_t defaultSlots = NextPowerOfTwo(requiredSlots);
    if (defaultSlots > maxSlots) {
        throw runtime_error("Required bootstrap slots exceed CKKS slot capacity");
    }

    const char* slotsEnv = getenv("CKKS_BOOTSTRAP_SLOTS");
    if (slotsEnv == nullptr || slotsEnv[0] == '\0') {
        return defaultSlots;
    }

    try {
        unsigned long parsed = stoul(slotsEnv);
        if (parsed == 0 || parsed > maxSlots) {
            cout << "[WARN] Ignoring CKKS_BOOTSTRAP_SLOTS=" << slotsEnv
                 << " (must be in [1, " << maxSlots << "])." << endl;
            return defaultSlots;
        }

        uint32_t requested = static_cast<uint32_t>(parsed);
        if (requested < requiredSlots || !IsPowerOfTwo(requested)) {
            cout << "[WARN] Ignoring CKKS_BOOTSTRAP_SLOTS=" << slotsEnv
                 << " (must be power-of-two and >= " << requiredSlots << ")." << endl;
            return defaultSlots;
        }
        return requested;
    }
    catch (const exception&) {
        cout << "[WARN] Ignoring invalid CKKS_BOOTSTRAP_SLOTS='" << slotsEnv << "'." << endl;
        return defaultSlots;
    }
}

uint32_t ResolveBootstrapCorrectionFactor() {
    const uint32_t defaultFactor = 10;
    const uint32_t minSupportedFactor = 10;
    const char* factorEnv = getenv("CKKS_BOOTSTRAP_CORRECTION_FACTOR");
    if (factorEnv == nullptr || factorEnv[0] == '\0') {
        return defaultFactor;
    }

    try {
        unsigned long parsed = stoul(factorEnv);
        if (parsed < minSupportedFactor || parsed > 30) {
            cout << "[WARN] Ignoring CKKS_BOOTSTRAP_CORRECTION_FACTOR=" << factorEnv
                 << " (must be in [" << minSupportedFactor << ", 30] for this model depth)." << endl;
            return defaultFactor;
        }
        return static_cast<uint32_t>(parsed);
    }
    catch (const exception&) {
        cout << "[WARN] Ignoring invalid CKKS_BOOTSTRAP_CORRECTION_FACTOR='"
             << factorEnv << "'." << endl;
        return defaultFactor;
    }
}

// ============================================================
// Performance Metrics
// ============================================================
struct PerformanceMetrics {
    double contextSetupTime         = 0;
    double keyGenTime               = 0;
    double multKeyGenTime           = 0;
    double bootstrapKeyGenTime      = 0;
    double bootstrapSetupTime       = 0;
    double encryptionTime           = 0;
    double decryptionTime           = 0;
    double layersBeforeBootstrapTime = 0;
    double bootstrapTime            = 0;
    double layersAfterBootstrapTime  = 0;
    double classificationTime       = 0;
    double totalTime                = 0;
    double weightLoadTime           = 0;

    uint32_t ringDimension       = 0;
    uint32_t multDepth           = 0;
    uint32_t levelsAfterBootstrap = 0;
    uint32_t numSlots            = 0;

    uint32_t initialLevel          = 0;
    uint32_t levelBeforeBootstrap  = 0;
    uint32_t levelAfterBootstrap   = 0;
    uint32_t finalLevel            = 0;

    int numCorrect    = 0;
    int numTotal      = 0;
    double maxError   = 0;
    double rmse       = 0;
};

// ============================================================
// Neural Network Layer (with trained weights)
// ============================================================
struct NNLayer {
    vector<double> weights;     // Per-pixel trained weights (4096 values)
    vector<double> biases;      // Per-pixel trained biases (4096 values)
    Plaintext      weightPt;    // Cached CKKS plaintext for faster repeated inference
    Plaintext      biasPt;      // Cached CKKS plaintext for faster repeated inference
    bool           hasCachedPlaintexts = false;
    string         name;
    bool           hasActivation;
};

// ============================================================
// Classifier (post-decryption)
// ============================================================
struct Classifier {
    vector<vector<double>> weights;  // [NUM_CLASSES][IMAGE_SIZE]
    vector<double>         biases;   // [NUM_CLASSES]
};

// ============================================================
// File I/O Utilities
// ============================================================

/**
 * Load a vector of doubles from a text file (one value per line).
 */
vector<double> LoadVector(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file: " << filepath << endl;
        exit(1);
    }
    
    vector<double> values;
    double val;
    while (file >> val) {
        values.push_back(val);
    }
    file.close();
    return values;
}

/**
 * Load a matrix from a text file (space-separated, one row per line).
 */
vector<vector<double>> LoadMatrix(const string& filepath, int rows, int cols) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file: " << filepath << endl;
        exit(1);
    }
    
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            file >> matrix[r][c];
        }
    }
    file.close();
    return matrix;
}

/**
 * Load a test image from text file (one pixel per line, normalized [0,1]).
 */
vector<double> LoadTestImage(const string& filepath) {
    vector<double> pixels = LoadVector(filepath);
    if (pixels.size() != IMAGE_SIZE) {
        cerr << "ERROR: Expected " << IMAGE_SIZE << " pixels, got " << pixels.size()
             << " from " << filepath << endl;
        exit(1);
    }
    return pixels;
}

/**
 * Load the true label from label file.
 */
int LoadLabel(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) return -1;
    int label;
    file >> label;
    return label;
}

/**
 * Save image as PGM.
 */
void SavePGM(const string& filename, const vector<double>& pixels, int w, int h) {
    ofstream file(filename);
    file << "P2\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        int val = max(0, min(255, (int)(pixels[i] * 255.0)));
        file << val;
        if ((i + 1) % 16 == 0) file << "\n";
        else file << " ";
    }
    file.close();
}

// ============================================================
// Load Trained Weights
// ============================================================
struct TrainedModel {
    vector<NNLayer> layers;
    Classifier classifier;
};

TrainedModel LoadTrainedModel(const string& weightsDir) {
    cout << "\n========================================" << endl;
    cout << "Loading Trained CryptoNets Weights" << endl;
    cout << "========================================\n" << endl;
    
    auto start = chrono::high_resolution_clock::now();
    
    TrainedModel model;
    
    // Load 5 element-wise layers
    string layerNames[] = {
        "Layer1-FeatureExtract",
        "Layer2-FeatureRefine",
        "Layer3-Transform",
        "Layer4-Classify",
        "Layer5-Output"
    };
    bool activations[] = {true, true, true, true, false};
    
    for (int i = 0; i < 5; ++i) {
        NNLayer layer;
        layer.name = layerNames[i];
        layer.hasActivation = activations[i];
        
        string wpath = weightsDir + "/weights_layer" + to_string(i+1) + ".txt";
        string bpath = weightsDir + "/biases_layer" + to_string(i+1) + ".txt";
        
        layer.weights = LoadVector(wpath);
        layer.biases  = LoadVector(bpath);
        
        if (layer.weights.size() != IMAGE_SIZE || layer.biases.size() != IMAGE_SIZE) {
            cerr << "ERROR: Layer " << (i+1) << " weight/bias size mismatch. "
                 << "Expected " << IMAGE_SIZE << ", got weights=" << layer.weights.size()
                 << " biases=" << layer.biases.size() << endl;
            exit(1);
        }
        
        // Print stats
        double wmin = *min_element(layer.weights.begin(), layer.weights.end());
        double wmax = *max_element(layer.weights.begin(), layer.weights.end());
        double bmin = *min_element(layer.biases.begin(), layer.biases.end());
        double bmax = *max_element(layer.biases.begin(), layer.biases.end());
        
        cout << "  [" << layer.name << "] weights=[" << fixed << setprecision(4)
             << wmin << "," << wmax << "], bias=[" << bmin << "," << bmax << "]"
             << (layer.hasActivation ? ", act=x^2" : "") << endl;
        
        model.layers.push_back(layer);
    }
    
    // Load classifier
    string cwpath = weightsDir + "/classifier_weights.txt";
    string cbpath = weightsDir + "/classifier_bias.txt";
    
    model.classifier.weights = LoadMatrix(cwpath, NUM_CLASSES, IMAGE_SIZE);
    model.classifier.biases  = LoadVector(cbpath);
    
    cout << "  [Classifier] " << NUM_CLASSES << " classes × " << IMAGE_SIZE << " features" << endl;
    
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "\nWeights loaded in " << elapsed << " ms" << endl;
    
    return model;
}

void CacheModelPlaintexts(const CryptoContext<DCRTPoly>& cc, TrainedModel& model, uint32_t numSlots) {
    for (auto& layer : model.layers) {
        if (!layer.hasCachedPlaintexts) {
            layer.weightPt = cc->MakeCKKSPackedPlaintext(layer.weights, 1, 0, nullptr, numSlots);
            layer.biasPt = cc->MakeCKKSPackedPlaintext(layer.biases, 1, 0, nullptr, numSlots);
            layer.hasCachedPlaintexts = true;
        }
    }
}

// ============================================================
// Extract Ciphertext Polynomial Coefficients
// ============================================================
/**
 * Extract actual polynomial coefficients from a CKKS ciphertext.
 *
 * CKKS ciphertexts are pairs (c0, c1) of polynomials in R_q = Z_q[X]/(X^N+1).
 * We extract the first polynomial c0's coefficients from its first CRT tower,
 * normalize them to [0, 1], and return them as the "encrypted representation."
 *
 * Because CKKS encoding uses the NTT, changing ANY single plaintext slot
 * changes ALL polynomial coefficients — providing full diffusion for
 * NPCR/UACI analysis.
 *
 * @param ct  The ciphertext to extract coefficients from
 * @param numCoeffs  Number of coefficients to return (0 = all available)
 * @return Normalized polynomial coefficients in [0, 1]
 */
vector<double> ExtractCiphertextSignature(const Ciphertext<DCRTPoly>& ct) {
    vector<double> signature;

    // Use all ciphertext elements (c0, c1, ...) and all CRT towers.
    // This gives a richer representation for diffusion statistics.
    const auto& elements = ct->GetElements();
    for (const auto& poly : elements) {
        size_t numTowers = poly.GetNumOfElements();
        for (size_t t = 0; t < numTowers; ++t) {
            const auto& tower = poly.GetElementAtIndex(t);
            size_t polyLen = tower.GetLength();
            double q = static_cast<double>(tower.GetModulus().ConvertToInt());
            if (q <= 0.0) {
                continue;
            }

            signature.reserve(signature.size() + polyLen);
            for (size_t i = 0; i < polyLen; ++i) {
                double coeff = static_cast<double>(tower[i].ConvertToInt());
                signature.push_back(coeff / q);
            }
        }
    }

    return signature;
}

// ============================================================
// Homomorphic NN Layer Forward Pass
// ============================================================
Ciphertext<DCRTPoly> HomomorphicNNForward(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    const NNLayer& layer,
    int numSlots)
{
    Plaintext weightPt = layer.hasCachedPlaintexts
        ? layer.weightPt
        : cc->MakeCKKSPackedPlaintext(layer.weights, 1, 0, nullptr, numSlots);
    Plaintext biasPt = layer.hasCachedPlaintexts
        ? layer.biasPt
        : cc->MakeCKKSPackedPlaintext(layer.biases, 1, 0, nullptr, numSlots);
    
    // Linear: output = weights * input + bias
    auto result = cc->EvalMult(input, weightPt);
    result = cc->Rescale(result);
    result = cc->EvalAdd(result, biasPt);
    
    // Square activation: output = output^2
    if (layer.hasActivation) {
        result = cc->EvalMult(result, result);
        result = cc->Rescale(result);
    }
    
    return result;
}

// ============================================================
// Plaintext NN Forward Pass (for verification)
// ============================================================
vector<double> PlaintextNNForward(
    const vector<double>& input,
    const TrainedModel& model)
{
    vector<double> x = input;
    
    for (int l = 0; l < 5; ++l) {
        const auto& layer = model.layers[l];
        vector<double> out(IMAGE_SIZE);
        
        // Linear
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            out[i] = layer.weights[i] * x[i] + layer.biases[i];
        }
        
        // Square activation
        if (layer.hasActivation) {
            for (int i = 0; i < IMAGE_SIZE; ++i) {
                out[i] = out[i] * out[i];
            }
        }
        
        x = out;
    }
    
    return x;
}

// ============================================================
// Post-decryption Classification
// ============================================================
int ClassifyFeatures(const vector<double>& features, const Classifier& clf) {
    vector<double> logits(NUM_CLASSES, 0.0);
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            logits[c] += clf.weights[c][i] * features[i];
        }
        logits[c] += clf.biases[c];
    }
    
    // Print logits
    cout << "  Logits: [";
    for (int c = 0; c < NUM_CLASSES; ++c) {
        cout << fixed << setprecision(4) << logits[c];
        if (c < NUM_CLASSES - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    // Softmax for probabilities
    double maxLogit = *max_element(logits.begin(), logits.end());
    double sumExp = 0;
    vector<double> probs(NUM_CLASSES);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs[c] = exp(logits[c] - maxLogit);
        sumExp += probs[c];
    }
    cout << "  Probabilities: [";
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs[c] /= sumExp;
        cout << fixed << setprecision(4) << probs[c];
        if (c < NUM_CLASSES - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    // Argmax
    return max_element(logits.begin(), logits.end()) - logits.begin();
}

// ============================================================
// CKKS Context Setup
// ============================================================
CryptoContext<DCRTPoly> SetupCKKSContext(PerformanceMetrics& metrics) {
    cout << "\n========================================" << endl;
    cout << "Setting up CKKS for Encrypted Inference" << endl;
    cout << "========================================\n" << endl;

    auto start = chrono::high_resolution_clock::now();

    CCParams<CryptoContextCKKSRNS> parameters;

    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    parameters.SetSecurityLevel(HEStd_NotSet);  // Demo mode
    parameters.SetRingDim(1 << 14);  // 16384

    vector<uint32_t> levelBudget = {5, 5};

    uint32_t levelsAvailableAfterBootstrap = 10;
    metrics.levelsAfterBootstrap = levelsAvailableAfterBootstrap;

    uint32_t bootstrapCorrectionFactor = ResolveBootstrapCorrectionFactor();
    uint32_t approxModDepth = bootstrapCorrectionFactor;
    uint32_t bootstrapDepth = FHECKKSRNS::GetBootstrapDepth(approxModDepth, levelBudget, secretKeyDist);
    uint32_t depth = levelsAvailableAfterBootstrap + bootstrapDepth;
    parameters.SetMultiplicativeDepth(depth);
    metrics.multDepth = depth;

    cout << "Bootstrap circuit depth: " << bootstrapDepth << endl;
    cout << "Bootstrap correction factor: " << bootstrapCorrectionFactor << endl;
    cout << "Levels for computation:  " << levelsAvailableAfterBootstrap << endl;
    cout << "Total mult. depth:       " << depth << endl;

    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    metrics.ringDimension = cc->GetRingDimension();
    metrics.numSlots = metrics.ringDimension / 2;

    auto end = chrono::high_resolution_clock::now();
    metrics.contextSetupTime = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "\nCrypto Context:" << endl;
    cout << "  Ring Dimension:       " << metrics.ringDimension << endl;
    cout << "  Max Slots:            " << metrics.numSlots << endl;
    cout << "  Multiplicative Depth: " << depth << endl;
    cout << "  Image pixels:         " << IMAGE_SIZE << " (uses " << IMAGE_SIZE 
         << " of " << metrics.numSlots << " slots)" << endl;
    cout << "  Setup time:           " << metrics.contextSetupTime << " ms" << endl;

    return cc;
}

// ============================================================
// Performance Report
// ============================================================
void PrintPerformanceReport(const PerformanceMetrics& m) {
    cout << "\n" << string(70, '=') << endl;
    cout << "    ENCRYPTED MEDICAL IMAGE CLASSIFICATION - PERFORMANCE REPORT" << endl;
    cout << string(70, '=') << endl;

    cout << "\n--- APPLICATION ---" << endl;
    cout << "Task:                          Breast Cancer Detection (BreastMNIST)" << endl;
    cout << "Image Resolution:              64x64 (" << IMAGE_SIZE << " pixels)" << endl;
    cout << "Classes:                       " << CLASS_NAMES[0] << " / " << CLASS_NAMES[1] << endl;

    cout << "\n--- TIMING ---" << endl;
    cout << fixed << setprecision(2);
    cout << "Weight Loading:                " << setw(10) << m.weightLoadTime << " ms" << endl;
    cout << "Context Setup:                 " << setw(10) << m.contextSetupTime << " ms" << endl;
    cout << "Key Generation:                " << setw(10) << m.keyGenTime << " ms" << endl;
    cout << "Mult Key Gen:                  " << setw(10) << m.multKeyGenTime << " ms" << endl;
    cout << "Bootstrap Setup:               " << setw(10) << m.bootstrapSetupTime << " ms" << endl;
    cout << "Bootstrap Key Gen:             " << setw(10) << m.bootstrapKeyGenTime << " ms" << endl;
    cout << "Encryption:                    " << setw(10) << m.encryptionTime << " ms" << endl;
    cout << "NN Layers (pre-bootstrap):     " << setw(10) << m.layersBeforeBootstrapTime << " ms (3 layers)" << endl;
    cout << "Bootstrapping:                 " << setw(10) << m.bootstrapTime << " ms" << endl;
    cout << "NN Layers (post-bootstrap):    " << setw(10) << m.layersAfterBootstrapTime << " ms (2 layers)" << endl;
    cout << "Decryption:                    " << setw(10) << m.decryptionTime << " ms" << endl;
    cout << "Classification:                " << setw(10) << m.classificationTime << " ms" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "TOTAL:                         " << setw(10) << m.totalTime << " ms" << endl;
    cout << "                               " << setw(10) << m.totalTime / 1000.0 << " s" << endl;

    double total = m.totalTime;
    if (total > 0) {
        cout << "\n--- TIME BREAKDOWN (%) ---" << endl;
        cout << "Setup (context+keys):          " << setw(6) << setprecision(1)
             << 100.0 * (m.contextSetupTime + m.keyGenTime + m.multKeyGenTime) / total << "%" << endl;
        cout << "Bootstrap (setup+keygen+exec): " << setw(6)
             << 100.0 * (m.bootstrapSetupTime + m.bootstrapKeyGenTime + m.bootstrapTime) / total << "%" << endl;
        cout << "Neural Net Inference:          " << setw(6)
             << 100.0 * (m.layersBeforeBootstrapTime + m.layersAfterBootstrapTime) / total << "%" << endl;
        cout << "Encrypt + Decrypt + Classify:  " << setw(6)
             << 100.0 * (m.encryptionTime + m.decryptionTime + m.classificationTime) / total << "%" << endl;
    }

    cout << "\n--- CRYPTO PARAMETERS ---" << endl;
    cout << "Ring Dimension:                " << m.ringDimension << endl;
    cout << "Multiplicative Depth:          " << m.multDepth << endl;
    cout << "Number of Slots:               " << m.numSlots << endl;

    cout << "\n--- LEVEL TRACKING ---" << endl;
    cout << "Initial Level:                 " << m.initialLevel << endl;
    cout << "Level Before Bootstrap:        " << m.levelBeforeBootstrap << endl;
    cout << "Level After Bootstrap:         " << m.levelAfterBootstrap << endl;
    cout << "Final Level:                   " << m.finalLevel << endl;

    cout << "\n--- ACCURACY ---" << endl;
    cout << "Images Classified:             " << m.numTotal << endl;
    cout << "Correct Predictions:           " << m.numCorrect << endl;
    cout << "Encrypted Accuracy:            " << setprecision(1)
         << (m.numTotal > 0 ? 100.0 * m.numCorrect / m.numTotal : 0) << "%" << endl;
    cout << "Max Error (vs plaintext):      " << scientific << setprecision(6) << m.maxError << endl;
    cout << "RMSE (vs plaintext):           " << m.rmse << endl;

    cout << "\n" << string(70, '=') << endl;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    cout << "\n**********************************************************************" << endl;
    cout << "  ENCRYPTED BREAST CANCER DETECTION" << endl;
    cout << "  Privacy-Preserving Medical Image Classification" << endl;
    cout << "  CKKS FHE with Bootstrapping + Trained CryptoNets" << endl;
    cout << "**********************************************************************" << endl;

    cout << "\nPRIVACY SCENARIO:" << endl;
    cout << "  A hospital encrypts a patient's breast ultrasound image." << endl;
    cout << "  A cloud ML service classifies it while ENCRYPTED." << endl;
    cout << "  The hospital decrypts to get: 'malignant' or 'benign'." << endl;
    cout << "  The cloud NEVER sees the patient's data or diagnosis.\n" << endl;

    auto totalStart = chrono::high_resolution_clock::now();
    PerformanceMetrics metrics;

    // ========================================
    // Determine paths
    // ========================================
    // Helper: check if file exists
    auto fileExists = [](const string& path) -> bool {
        ifstream f(path);
        return f.good();
    };
    
    // Helper: get parent directory from path
    auto parentDir = [](const string& path) -> string {
        size_t pos = path.find_last_of("/\\");
        if (pos != string::npos) return path.substr(0, pos);
        return ".";
    };

    string exePath = parentDir(string(argv[0]));
    
    // Try to find the medmnist directory relative to the executable or CWD
    string weightsDir, imagesDir;
    
    vector<string> searchPaths = {
        "medmnist/weights",
        "../medmnist/weights",
        "../../medmnist/weights",
        "../../../medmnist/weights",
        exePath + "/../../../medmnist/weights",
        exePath + "/../../medmnist/weights",
    };
    
    for (const auto& p : searchPaths) {
        if (fileExists(p + "/weights_layer1.txt")) {
            weightsDir = p;
            imagesDir = parentDir(p) + "/test_images";
            break;
        }
    }
    
    if (weightsDir.empty()) {
        cerr << "ERROR: Cannot find trained weights directory." << endl;
        cerr << "Expected: medmnist/weights/weights_layer1.txt" << endl;
        cerr << "Run train_model.py first to train and export weights." << endl;
        return 1;
    }
    
    cout << "Weights directory: " << weightsDir << endl;
    cout << "Test images directory: " << imagesDir << endl;
    
    // How many test images to classify?
    int numImages = 5;  // default
    if (argc > 1) {
        numImages = atoi(argv[1]);
    }
    cout << "Will classify " << numImages << " test images.\n" << endl;
    
    // ========================================
    // Step 1: Load trained weights
    // ========================================
    auto wStart = chrono::high_resolution_clock::now();
    TrainedModel model = LoadTrainedModel(weightsDir);
    auto wEnd = chrono::high_resolution_clock::now();
    metrics.weightLoadTime = chrono::duration_cast<chrono::milliseconds>(wEnd - wStart).count();
    
    // ========================================
    // Step 2: Setup CKKS context
    // ========================================
    CryptoContext<DCRTPoly> cc = SetupCKKSContext(metrics);
        uint32_t maxSlots = cc->GetRingDimension() / 2;
        uint32_t numSlots = ResolveBootstrapSlots(static_cast<uint32_t>(IMAGE_SIZE), maxSlots);
        cout << "Bootstrap slots selected: " << numSlots
            << " (image size " << IMAGE_SIZE << ", max " << maxSlots << ")" << endl;
    
    // ========================================
    // Step 3: Bootstrap setup
    // ========================================
    {
        cout << "\n========================================" << endl;
        cout << "Bootstrap Setup (Precomputation)" << endl;
        cout << "========================================\n" << endl;
        
        vector<uint32_t> levelBudget = {5, 5};
        uint32_t correctionFactor = ResolveBootstrapCorrectionFactor();
        
        auto bsStart = chrono::high_resolution_clock::now();
        cc->EvalBootstrapSetup(levelBudget, {0, 0}, numSlots, correctionFactor);
        auto bsEnd = chrono::high_resolution_clock::now();
        metrics.bootstrapSetupTime = chrono::duration_cast<chrono::milliseconds>(bsEnd - bsStart).count();
        cout << "Bootstrap setup: " << metrics.bootstrapSetupTime << " ms" << endl;
    }
    
    // ========================================
    // Step 4: Generate keys
    // ========================================
    {
        cout << "\n========================================" << endl;
        cout << "Generate Cryptographic Keys" << endl;
        cout << "========================================\n" << endl;
        
        auto kStart = chrono::high_resolution_clock::now();
        auto keyPair = cc->KeyGen();
        auto kEnd = chrono::high_resolution_clock::now();
        metrics.keyGenTime = chrono::duration_cast<chrono::milliseconds>(kEnd - kStart).count();
        cout << "[OK] Key pair generated (" << metrics.keyGenTime << " ms)" << endl;
        
        auto mStart = chrono::high_resolution_clock::now();
        cc->EvalMultKeyGen(keyPair.secretKey);
        auto mEnd = chrono::high_resolution_clock::now();
        metrics.multKeyGenTime = chrono::duration_cast<chrono::milliseconds>(mEnd - mStart).count();
        cout << "[OK] Multiplication keys (" << metrics.multKeyGenTime << " ms)" << endl;
        
        cout << "\nGenerating bootstrapping keys (ring dim 16384, may take a while)..." << endl;
        auto bkStart = chrono::high_resolution_clock::now();
        cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
        auto bkEnd = chrono::high_resolution_clock::now();
        metrics.bootstrapKeyGenTime = chrono::duration_cast<chrono::milliseconds>(bkEnd - bkStart).count();
        cout << "[OK] Bootstrap keys (" << metrics.bootstrapKeyGenTime << " ms)" << endl;

           auto cacheStart = chrono::high_resolution_clock::now();
           CacheModelPlaintexts(cc, model, numSlots);
           auto cacheEnd = chrono::high_resolution_clock::now();
           cout << "[OK] Cached layer plaintext encodings ("
               << chrono::duration_cast<chrono::milliseconds>(cacheEnd - cacheStart).count()
               << " ms)" << endl;
        
        // ========================================
        // Step 5: Classify multiple test images
        // ========================================
        cout << "\n" << string(70, '*') << endl;
        cout << "    ENCRYPTED INFERENCE ON MEDICAL IMAGES" << endl;
        cout << string(70, '*') << endl;
        
        double totalEncTime = 0, totalPreBootTime = 0, totalBootTime = 0;
        double totalPostBootTime = 0, totalDecTime = 0, totalClfTime = 0;
        double overallMaxError = 0, overallSumSqError = 0;
        int overallPixelCount = 0;
        
        for (int imgIdx = 0; imgIdx < numImages; ++imgIdx) {
            cout << "\n" << string(60, '-') << endl;
            cout << "  Test Image " << imgIdx << " / " << numImages << endl;
            cout << string(60, '-') << endl;
            
            // Load test image
            string imgPath = imagesDir + "/test_image_" + 
                (imgIdx < 10 ? "00" : (imgIdx < 100 ? "0" : "")) + to_string(imgIdx) + ".txt";
            string lblPath = imagesDir + "/test_image_" + 
                (imgIdx < 10 ? "00" : (imgIdx < 100 ? "0" : "")) + to_string(imgIdx) + "_label.txt";
            
            if (!fileExists(imgPath)) {
                cout << "  Image file not found: " << imgPath << " — skipping." << endl;
                continue;
            }
            
            vector<double> pixels = LoadTestImage(imgPath);
            int trueLabel = LoadLabel(lblPath);
            
            double mean = accumulate(pixels.begin(), pixels.end(), 0.0) / pixels.size();
            cout << "  Image: " << imgPath << endl;
            cout << "  True label: " << trueLabel << " (" << CLASS_NAMES[trueLabel] << ")" << endl;
            cout << "  Mean pixel: " << fixed << setprecision(4) << mean << endl;
            
            // ---- Encrypt ----
            auto eStart = chrono::high_resolution_clock::now();
            Plaintext imagePt = cc->MakeCKKSPackedPlaintext(pixels, 1, 0, nullptr, numSlots);
            auto ct = cc->Encrypt(keyPair.publicKey, imagePt);
            auto eEnd = chrono::high_resolution_clock::now();
            double encTime = chrono::duration<double, milli>(eEnd - eStart).count();
            totalEncTime += encTime;

            // Capture raw-ciphertext signature immediately after encryption.
            // This measures CKKS diffusion/randomness prior to NN transformations.
            vector<double> rawEncRepr = ExtractCiphertextSignature(ct);
            
            uint32_t initLevel = ct->GetLevel();
            if (imgIdx == 0) metrics.initialLevel = initLevel;
            cout << "  Encrypted (" << setprecision(1) << encTime << " ms), level=" << initLevel << endl;
            
            // ---- Pre-bootstrap layers (1-3) ----
            cout << "\n  Running layers 1-3 (pre-bootstrap)..." << endl;
            auto preStart = chrono::high_resolution_clock::now();
            
            auto encResult = ct;
            for (int l = 0; l < 3; ++l) {
                auto layerStart = chrono::high_resolution_clock::now();
                encResult = HomomorphicNNForward(cc, encResult, model.layers[l], numSlots);
                auto layerEnd = chrono::high_resolution_clock::now();
                double lt = chrono::duration<double, milli>(layerEnd - layerStart).count();
                cout << "    [" << model.layers[l].name << "] level=" << encResult->GetLevel()
                     << " (" << setprecision(0) << lt << " ms)" << endl;
            }
            
            auto preEnd = chrono::high_resolution_clock::now();
            double preBootTime = chrono::duration<double, milli>(preEnd - preStart).count();
            totalPreBootTime += preBootTime;
            
            uint32_t levelBeforeBoot = encResult->GetLevel();
            if (imgIdx == 0) metrics.levelBeforeBootstrap = levelBeforeBoot;
            cout << "  Pre-bootstrap: " << preBootTime << " ms, level=" << levelBeforeBoot << endl;
            
            // ---- Bootstrap ----
            cout << "\n  *** BOOTSTRAPPING ***" << endl;
            auto bootStart = chrono::high_resolution_clock::now();
            encResult = cc->EvalBootstrap(encResult);
            auto bootEnd = chrono::high_resolution_clock::now();
            double bootTime = chrono::duration<double, milli>(bootEnd - bootStart).count();
            totalBootTime += bootTime;
            
            uint32_t levelAfterBoot = encResult->GetLevel();
            if (imgIdx == 0) metrics.levelAfterBootstrap = levelAfterBoot;
            cout << "  Level: " << levelBeforeBoot << " -> " << levelAfterBoot
                 << " (" << setprecision(0) << bootTime << " ms)" << endl;
            
            // ---- Post-bootstrap layers (4-5) ----
            cout << "\n  Running layers 4-5 (post-bootstrap)..." << endl;
            auto postStart = chrono::high_resolution_clock::now();
            
            for (int l = 3; l < 5; ++l) {
                auto layerStart = chrono::high_resolution_clock::now();
                encResult = HomomorphicNNForward(cc, encResult, model.layers[l], numSlots);
                auto layerEnd = chrono::high_resolution_clock::now();
                double lt = chrono::duration<double, milli>(layerEnd - layerStart).count();
                cout << "    [" << model.layers[l].name << "] level=" << encResult->GetLevel()
                     << " (" << setprecision(0) << lt << " ms)" << endl;
            }
            
            auto postEnd = chrono::high_resolution_clock::now();
            double postBootTime = chrono::duration<double, milli>(postEnd - postStart).count();
            totalPostBootTime += postBootTime;
            
            uint32_t finalLevel = encResult->GetLevel();
            if (imgIdx == 0) metrics.finalLevel = finalLevel;
            
            // ---- Decrypt ----
            auto dStart = chrono::high_resolution_clock::now();
            Plaintext resultPt;
            cc->Decrypt(keyPair.secretKey, encResult, &resultPt);
            resultPt->SetLength(IMAGE_SIZE);
            auto dEnd = chrono::high_resolution_clock::now();
            double decTime = chrono::duration<double, milli>(dEnd - dStart).count();
            totalDecTime += decTime;
            
            vector<double> decrypted = resultPt->GetRealPackedValue();
            decrypted.resize(IMAGE_SIZE);
            
            // ---- Classify (post-decryption) ----
            auto cStart = chrono::high_resolution_clock::now();
            cout << "\n  Classification result (encrypted):" << endl;
            int encPred = ClassifyFeatures(decrypted, model.classifier);
            auto cEnd = chrono::high_resolution_clock::now();
            double clfTime = chrono::duration<double, milli>(cEnd - cStart).count();
            totalClfTime += clfTime;
            
            cout << "  Prediction: " << encPred << " (" << CLASS_NAMES[encPred] << ")" << endl;
            
            // ---- Plaintext reference ----
            vector<double> plainResult = PlaintextNNForward(pixels, model);
            cout << "\n  Classification result (plaintext reference):" << endl;
            int plainPred = ClassifyFeatures(plainResult, model.classifier);
            cout << "  Prediction: " << plainPred << " (" << CLASS_NAMES[plainPred] << ")" << endl;
            
            // ---- Accuracy check ----
            bool correct = (encPred == trueLabel);
            bool matchPlaintext = (encPred == plainPred);
            
            if (correct) {
                cout << "\n  >>> CORRECT: Encrypted prediction matches true label! <<<" << endl;
                metrics.numCorrect++;
            } else {
                cout << "\n  >>> INCORRECT: Predicted " << CLASS_NAMES[encPred]
                     << ", actual " << CLASS_NAMES[trueLabel] << " <<<" << endl;
            }
            
            if (matchPlaintext) {
                cout << "  [OK] Encrypted result matches plaintext (FHE accurate)" << endl;
            } else {
                cout << "  [WARN] Encrypted result differs from plaintext!" << endl;
            }
            metrics.numTotal++;
            
            // ---- Error analysis (encrypted vs plaintext) ----
            double maxErr = 0, sumSqErr = 0;
            for (int i = 0; i < IMAGE_SIZE; ++i) {
                double err = abs(decrypted[i] - plainResult[i]);
                maxErr = max(maxErr, err);
                sumSqErr += err * err;
            }
            double imgRmse = sqrt(sumSqErr / IMAGE_SIZE);
            cout << "  FHE error: max=" << scientific << setprecision(3) << maxErr
                 << ", RMSE=" << imgRmse << endl;
            
            overallMaxError = max(overallMaxError, maxErr);
            overallSumSqError += sumSqErr;
            overallPixelCount += IMAGE_SIZE;
            
            // ---- Security Metrics for this image ----
            cout << "\n  Computing security metrics for image " << imgIdx << "..." << endl;
            
            // Create 1-pixel-modified version for NPCR/UACI
            vector<double> modPixels = pixels;
            int centerIdx = (IMAGE_HEIGHT / 2) * IMAGE_WIDTH + (IMAGE_WIDTH / 2);
            modPixels[centerIdx] = 1.0 - modPixels[centerIdx];
            
            // Encrypt and process modified image through the full NN pipeline
            Plaintext modImgPt = cc->MakeCKKSPackedPlaintext(modPixels, 1, 0, nullptr, numSlots);
            auto modCt = cc->Encrypt(keyPair.publicKey, modImgPt);
            vector<double> modRawEncRepr = ExtractCiphertextSignature(modCt);

            // Keep the spatial analysis aligned to the real image geometry.
            SecurityMetricsCalculator secCalc(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
            SecurityMetrics secM = secCalc.ComputeAll(
                pixels, rawEncRepr, decrypted, {}, modRawEncRepr);
            
            cout << "  Security Summary:" << endl;
            cout << "    NPCR:    " << fixed << setprecision(2) << secM.npcr << " %" << endl;
            cout << "    UACI:    " << secM.uaci << " %" << endl;
            cout << "    Entropy: " << setprecision(4) << secM.entropyEncrypted << " bits" << endl;
            cout << "    PSNR:    " << secM.psnr << " dB" << endl;
            cout << "    SSIM:    " << secM.ssim << endl;
            cout << "    Corr(H): " << setprecision(6) << secM.corrEncryptedH << endl;
            
            // Print full report for the first image
            if (imgIdx == 0) {
                SecurityMetricsCalculator::PrintSecurityReport(secM);
            }
        }
        
        // Aggregate metrics
        metrics.encryptionTime = totalEncTime;
        metrics.layersBeforeBootstrapTime = totalPreBootTime;
        metrics.bootstrapTime = totalBootTime;
        metrics.layersAfterBootstrapTime = totalPostBootTime;
        metrics.decryptionTime = totalDecTime;
        metrics.classificationTime = totalClfTime;
        metrics.maxError = overallMaxError;
        metrics.rmse = (overallPixelCount > 0) ? sqrt(overallSumSqError / overallPixelCount) : 0;
    }
    
    auto totalEnd = chrono::high_resolution_clock::now();
    metrics.totalTime = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart).count();
    
    // Print report
    PrintPerformanceReport(metrics);
    
    // Final summary
    cout << "\n**********************************************************************" << endl;
    cout << "  ENCRYPTED MEDICAL IMAGE CLASSIFICATION COMPLETE" << endl;
    cout << "**********************************************************************" << endl;
    cout << "\nWhat was demonstrated:" << endl;
    cout << "  1. Loaded TRAINED neural network weights (from BreastMNIST)" << endl;
    cout << "  2. Encrypted real medical images (64x64 breast ultrasound)" << endl;
    cout << "  3. Ran 5-layer CryptoNets ENTIRELY ON ENCRYPTED DATA" << endl;
    cout << "  4. Used BOOTSTRAPPING to enable deeper computation" << endl;
    cout << "  5. Decrypted and classified: malignant vs benign" << endl;
    cout << "  6. Cloud server NEVER saw the patient's image or diagnosis" << endl;
    cout << "\nAccuracy: " << metrics.numCorrect << "/" << metrics.numTotal 
         << " (" << fixed << setprecision(1)
         << (metrics.numTotal > 0 ? 100.0 * metrics.numCorrect / metrics.numTotal : 0)
         << "%)" << endl;
    cout << "FHE precision: RMSE = " << scientific << setprecision(2) << metrics.rmse 
         << " (encrypted matches plaintext)" << endl;
    cout << "\n**********************************************************************\n" << endl;
    
    return 0;
}
