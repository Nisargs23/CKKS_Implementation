/**
 * Encrypted Medical Image Classification — RGB Support
 * ======================================================
 *
 * Extends the BreastMNIST pipeline to support RGB (3-channel) datasets:
 *   - PathMNIST (colon pathology, 9 classes)
 *   - DermaMNIST (dermatoscopy, 7 classes)
 *   - BloodMNIST (blood cells, 8 classes)
 *   - RetinaMNIST (retinal OCT, 5 classes)
 *
 * RGB ENCRYPTION STRATEGY:
 *   Since 3×4096 = 12288 > 8192 (max CKKS slots at ring dim 16384),
 *   each RGB channel is encrypted as a separate ciphertext:
 *     ct_R = Encrypt(red_channel)     // 4096 slots
 *     ct_G = Encrypt(green_channel)   // 4096 slots
 *     ct_B = Encrypt(blue_channel)    // 4096 slots
 *
 *   Each channel is processed through the NN independently with its
 *   own per-channel weights. After decryption, features from all 3
 *   channels are concatenated for the classification step.
 *
 *   This is the standard approach in FHE image processing literature.
 *
 * Also supports greyscale (1 channel) — auto-detected from model_info.txt.
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
#include <cctype>
#include <cstdlib>
#include <filesystem>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "openfhe/pke/openfhe.h"
#include "openfhe/pke/cryptocontext-ser.h"
#include "openfhe/pke/key/key-ser.h"
#include "openfhe/pke/scheme/ckksrns/ckksrns-fhe.h"
#include "security_metrics.h"

using namespace lbcrypto;
using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Constants (may be overridden from model_info.txt)
// ============================================================
const int IMAGE_WIDTH  = 64;
const int IMAGE_HEIGHT = 64;
const int CHANNEL_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;  // 4096 pixels per channel

enum class ChannelExecutionMode {
    Auto,
    OuterParallel,
    Sequential,
};

ChannelExecutionMode ParseChannelExecutionMode(const string& rawMode) {
    string mode = rawMode;
    transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(tolower(c));
    });

    if (mode == "parallel" || mode == "outer") {
        return ChannelExecutionMode::OuterParallel;
    }
    if (mode == "sequential" || mode == "seq") {
        return ChannelExecutionMode::Sequential;
    }
    return ChannelExecutionMode::Auto;
}

string ChannelExecutionModeName(ChannelExecutionMode mode) {
    if (mode == ChannelExecutionMode::OuterParallel) {
        return "parallel";
    }
    if (mode == ChannelExecutionMode::Sequential) {
        return "sequential";
    }
    return "auto";
}

bool ShouldUseOuterChannelParallel(ChannelExecutionMode mode, int numChannels) {
    if (numChannels <= 1) {
        return false;
    }

    if (mode == ChannelExecutionMode::OuterParallel) {
        return true;
    }
    if (mode == ChannelExecutionMode::Sequential) {
        return false;
    }

#ifdef _OPENMP
    // Auto mode: prioritize channel-level wall-clock speedup when threading is
    // available. This remains the best default for the current RGB pipeline.
    int maxThreads = omp_get_max_threads();
    return maxThreads > 1;
#else
    return false;
#endif
}

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
        cout << "[WARN] Ignoring invalid CKKS_BOOTSTRAP_SLOTS='" << slotsEnv
             << "'." << endl;
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

bool ParseBoolEnv(const char* raw, bool defaultValue = false) {
    if (raw == nullptr || raw[0] == '\0') {
        return defaultValue;
    }

    string value(raw);
    transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(tolower(c));
    });

    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return defaultValue;
}

string ResolveCacheBaseDir() {
    const char* cacheDirEnv = getenv("CKKS_CACHE_DIR");
    if (cacheDirEnv != nullptr && cacheDirEnv[0] != '\0') {
        return string(cacheDirEnv);
    }
    return "final/cache/ckks_rgb";
}

struct KeyCachePaths {
    string profileDir;
    string contextPath;
    string publicKeyPath;
    string secretKeyPath;
    string evalMultKeyPath;
    string evalAutoKeyPath;
};

KeyCachePaths BuildKeyCachePaths(const string& baseDir, uint32_t ringDim, uint32_t numSlots,
                                 uint32_t correctionFactor) {
    fs::path profileDir = fs::path(baseDir) /
        ("ring" + to_string(ringDim) + "_slots" + to_string(numSlots) + "_cf" + to_string(correctionFactor));

    KeyCachePaths paths;
    paths.profileDir = profileDir.string();
    paths.contextPath = (profileDir / "context.bin").string();
    paths.publicKeyPath = (profileDir / "public_key.bin").string();
    paths.secretKeyPath = (profileDir / "secret_key.bin").string();
    paths.evalMultKeyPath = (profileDir / "eval_mult_keys.bin").string();
    paths.evalAutoKeyPath = (profileDir / "eval_auto_keys.bin").string();
    return paths;
}

bool SaveCryptoCache(const KeyCachePaths& paths, const CryptoContext<DCRTPoly>& cc,
                     const KeyPair<DCRTPoly>& keyPair) {
    try {
        error_code ec;
        fs::create_directories(paths.profileDir, ec);
        if (ec) {
            cout << "[WARN] Failed to create key cache directory: " << paths.profileDir << endl;
            return false;
        }

        if (!Serial::SerializeToFile(paths.contextPath, cc, SerType::BINARY)) {
            return false;
        }
        if (!Serial::SerializeToFile(paths.publicKeyPath, keyPair.publicKey, SerType::BINARY)) {
            return false;
        }
        if (!Serial::SerializeToFile(paths.secretKeyPath, keyPair.secretKey, SerType::BINARY)) {
            return false;
        }

        ofstream evalMultOut(paths.evalMultKeyPath, ios::binary | ios::out);
        if (!evalMultOut.is_open() || !cc->SerializeEvalMultKey(evalMultOut, SerType::BINARY, cc)) {
            return false;
        }
        evalMultOut.close();

        ofstream evalAutoOut(paths.evalAutoKeyPath, ios::binary | ios::out);
        if (!evalAutoOut.is_open() || !cc->SerializeEvalAutomorphismKey(evalAutoOut, SerType::BINARY, cc)) {
            return false;
        }
        evalAutoOut.close();

        return true;
    }
    catch (const exception& ex) {
        cout << "[WARN] Cache serialization failed: " << ex.what() << endl;
        return false;
    }
}

bool LoadCryptoCache(const KeyCachePaths& paths, CryptoContext<DCRTPoly>& cc,
                     KeyPair<DCRTPoly>& keyPair, string& reason) {
    if (!fs::exists(paths.contextPath) || !fs::exists(paths.publicKeyPath) ||
        !fs::exists(paths.secretKeyPath) || !fs::exists(paths.evalMultKeyPath) ||
        !fs::exists(paths.evalAutoKeyPath)) {
        reason = "cache files missing";
        return false;
    }

    try {
        CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
        CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys();
        CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys();
        CryptoContextImpl<DCRTPoly>::ClearEvalSumKeys();

        if (!Serial::DeserializeFromFile(paths.contextPath, cc, SerType::BINARY)) {
            reason = "context deserialize failed";
            return false;
        }
        if (!Serial::DeserializeFromFile(paths.publicKeyPath, keyPair.publicKey, SerType::BINARY)) {
            reason = "public key deserialize failed";
            return false;
        }
        if (!Serial::DeserializeFromFile(paths.secretKeyPath, keyPair.secretKey, SerType::BINARY)) {
            reason = "secret key deserialize failed";
            return false;
        }

        ifstream evalMultIn(paths.evalMultKeyPath, ios::binary | ios::in);
        if (!evalMultIn.is_open() || !cc->DeserializeEvalMultKey(evalMultIn, SerType::BINARY)) {
            reason = "eval mult key deserialize failed";
            return false;
        }
        evalMultIn.close();

        ifstream evalAutoIn(paths.evalAutoKeyPath, ios::binary | ios::in);
        if (!evalAutoIn.is_open() || !cc->DeserializeEvalAutomorphismKey(evalAutoIn, SerType::BINARY)) {
            reason = "eval automorphism key deserialize failed";
            return false;
        }
        evalAutoIn.close();

        reason.clear();
        return true;
    }
    catch (const exception& ex) {
        reason = ex.what();
        return false;
    }
}

// ============================================================
// Model Info (loaded from model_info.txt)
// ============================================================
struct ModelInfo {
    string dataset    = "breastmnist";
    int numChannels   = 1;       // 1 = greyscale, 3 = RGB
    int imageSize     = 4096;    // total pixels: channels * width * height
    int channelSize   = 4096;
    int numClasses    = 2;
    int numLayers     = 5;
    string colorMode  = "greyscale";
};

ModelInfo LoadModelInfo(const string& path) {
    ModelInfo info;
    ifstream f(path);
    if (!f.is_open()) {
        cout << "Warning: Cannot open " << path << ", using defaults." << endl;
        return info;
    }
    string line;
    while (getline(f, line)) {
        auto eq = line.find('=');
        if (eq == string::npos) continue;
        string key = line.substr(0, eq);
        string val = line.substr(eq + 1);
        if (key == "dataset")       info.dataset = val;
        if (key == "num_channels")  info.numChannels = stoi(val);
        if (key == "image_size")    info.imageSize = stoi(val);
        if (key == "channel_size")  info.channelSize = stoi(val);
        if (key == "num_classes")   info.numClasses = stoi(val);
        if (key == "num_layers")    info.numLayers = stoi(val);
        if (key == "color_mode")    info.colorMode = val;
    }
    return info;
}

// ============================================================
// Neural Network Layer (per-channel weights)
// ============================================================
struct NNLayerChannel {
    vector<double> weights;   // [CHANNEL_SIZE] weights for this channel
    vector<double> biases;    // [CHANNEL_SIZE] biases for this channel
    Plaintext      weightPt;  // Cached CKKS plaintext for faster repeated inference
    Plaintext      biasPt;    // Cached CKKS plaintext for faster repeated inference
    bool           hasCachedPlaintexts = false;
    string         name;
    bool           hasActivation;
};

struct NNLayer {
    vector<NNLayerChannel> channels;  // One per channel (1 for grey, 3 for RGB)
    string name;
    bool   hasActivation;
};

// ============================================================
// Classifier (post-decryption, cross-channel)
// ============================================================
struct Classifier {
    vector<vector<double>> weights;  // [numClasses][imageSize]
    vector<double>         biases;   // [numClasses]
    int numClasses;
    int imageSize;
};

struct RgbTimingMetrics {
    double weightLoadMs = 0.0;
    double contextSetupMs = 0.0;
    double bootstrapSetupMs = 0.0;
    double keyGenMs = 0.0;
    double multKeyGenMs = 0.0;
    double bootstrapKeyGenMs = 0.0;

    double encryptionMs = 0.0;
    double preBootstrapLayersMs = 0.0;
    double bootstrapEvalMs = 0.0;
    double postBootstrapLayersMs = 0.0;
    double decryptionMs = 0.0;
    double channelPipelineWallMs = 0.0;
    double classificationMs = 0.0;
    double plaintextReferenceMs = 0.0;
    double securityMetricsMs = 0.0;
    double totalMs = 0.0;
};

void PrintTimingReport(const RgbTimingMetrics& t) {
    cout << "\n" << string(70, '=') << endl;
    cout << "               RGB PIPELINE TIMING REPORT" << endl;
    cout << string(70, '=') << endl;
    cout << fixed << setprecision(2);
    cout << "Weight Loading:                " << setw(10) << t.weightLoadMs << " ms" << endl;
    cout << "Context Setup:                 " << setw(10) << t.contextSetupMs << " ms" << endl;
    cout << "Bootstrap Setup:               " << setw(10) << t.bootstrapSetupMs << " ms" << endl;
    cout << "Key Generation:                " << setw(10) << t.keyGenMs << " ms" << endl;
    cout << "Mult Key Gen:                  " << setw(10) << t.multKeyGenMs << " ms" << endl;
    cout << "Bootstrap Key Gen:             " << setw(10) << t.bootstrapKeyGenMs << " ms" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "Encryption (sum ch):           " << setw(10) << t.encryptionMs << " ms" << endl;
    cout << "Layers 1-3 (sum ch):           " << setw(10) << t.preBootstrapLayersMs << " ms" << endl;
    cout << "Bootstrapping (sum ch):        " << setw(10) << t.bootstrapEvalMs << " ms" << endl;
    cout << "Layers 4-5 (sum ch):           " << setw(10) << t.postBootstrapLayersMs << " ms" << endl;
    cout << "Decryption (sum ch):           " << setw(10) << t.decryptionMs << " ms" << endl;
    cout << "Channel Pipeline (wall):       " << setw(10) << t.channelPipelineWallMs << " ms" << endl;
    cout << "Classification:                " << setw(10) << t.classificationMs << " ms" << endl;
    cout << "Plaintext Reference:           " << setw(10) << t.plaintextReferenceMs << " ms" << endl;
    cout << "Security Metrics:              " << setw(10) << t.securityMetricsMs << " ms" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "TOTAL:                         " << setw(10) << t.totalMs << " ms" << endl;
    cout << "                               " << setw(10) << t.totalMs / 1000.0 << " s" << endl;
    cout << string(70, '=') << endl;
}

// ============================================================
// File I/O
// ============================================================
vector<double> LoadVector(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file: " << filepath << endl;
        exit(1);
    }
    vector<double> values;
    double val;
    while (file >> val) values.push_back(val);
    return values;
}

vector<double> ExtractCiphertextSignature(const Ciphertext<DCRTPoly>& ct) {
    vector<double> signature;

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

vector<vector<double>> LoadMatrix(const string& filepath, int rows, int cols) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file: " << filepath << endl;
        exit(1);
    }
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            file >> matrix[r][c];
    return matrix;
}

// ============================================================
// Load Trained Model (supports per-channel weights for RGB)
// ============================================================
struct TrainedModel {
    vector<NNLayer> layers;
    Classifier classifier;
    ModelInfo info;
};

TrainedModel LoadTrainedModel(const string& weightsDir, const ModelInfo& info) {
    cout << "\n========================================" << endl;
    cout << "Loading Trained CryptoNets Weights ("
         << (info.numChannels == 3 ? "RGB" : "Greyscale") << ")" << endl;
    cout << "========================================\n" << endl;

    TrainedModel model;
    model.info = info;

    string layerNames[] = {
        "Layer1-FeatureExtract", "Layer2-FeatureRefine", "Layer3-Transform",
        "Layer4-Classify", "Layer5-Output"
    };
    bool activations[] = {true, true, true, true, false};

    for (int i = 0; i < 5; ++i) {
        NNLayer layer;
        layer.name = layerNames[i];
        layer.hasActivation = activations[i];
        layer.channels.resize(info.numChannels);

        for (int ch = 0; ch < info.numChannels; ++ch) {
            string wpath, bpath;
            if (info.numChannels > 1) {
                // Per-channel files
                wpath = weightsDir + "/weights_layer" + to_string(i + 1) + "_ch" + to_string(ch) + ".txt";
                bpath = weightsDir + "/biases_layer" + to_string(i + 1) + "_ch" + to_string(ch) + ".txt";
            } else {
                wpath = weightsDir + "/weights_layer" + to_string(i + 1) + ".txt";
                bpath = weightsDir + "/biases_layer" + to_string(i + 1) + ".txt";
            }

            layer.channels[ch].weights = LoadVector(wpath);
            layer.channels[ch].biases  = LoadVector(bpath);
            layer.channels[ch].name = layerNames[i] + (info.numChannels > 1 ?
                string("_ch") + to_string(ch) : "");
            layer.channels[ch].hasActivation = activations[i];

            if ((int)layer.channels[ch].weights.size() != CHANNEL_SIZE) {
                cerr << "ERROR: Layer " << (i + 1) << " ch" << ch 
                     << " weight size = " << layer.channels[ch].weights.size()
                     << ", expected " << CHANNEL_SIZE << endl;
                exit(1);
            }
        }

        // Print stats
        double wmin = layer.channels[0].weights[0], wmax = wmin;
        for (int ch = 0; ch < info.numChannels; ++ch) {
            for (double w : layer.channels[ch].weights) {
                wmin = min(wmin, w);
                wmax = max(wmax, w);
            }
        }
        cout << "  [" << layer.name << "] channels=" << info.numChannels
             << ", weights=[" << fixed << setprecision(4) << wmin << "," << wmax << "]"
             << (layer.hasActivation ? ", act=x^2" : "") << endl;

        model.layers.push_back(layer);
    }

    // Load classifier (full cross-channel)
    string cwpath = weightsDir + "/classifier_weights.txt";
    string cbpath = weightsDir + "/classifier_bias.txt";

    model.classifier.weights = LoadMatrix(cwpath, info.numClasses, info.imageSize);
    model.classifier.biases  = LoadVector(cbpath);
    model.classifier.numClasses = info.numClasses;
    model.classifier.imageSize = info.imageSize;

    cout << "  [Classifier] " << info.numClasses << " classes x " 
         << info.imageSize << " features" << endl;

    return model;
}

void CacheModelPlaintexts(const CryptoContext<DCRTPoly>& cc, TrainedModel& model, uint32_t numSlots) {
    for (auto& layer : model.layers) {
        for (auto& channel : layer.channels) {
            if (!channel.hasCachedPlaintexts) {
                channel.weightPt = cc->MakeCKKSPackedPlaintext(channel.weights, 1, 0, nullptr, numSlots);
                channel.biasPt = cc->MakeCKKSPackedPlaintext(channel.biases, 1, 0, nullptr, numSlots);
                channel.hasCachedPlaintexts = true;
            }
        }
    }
}

// ============================================================
// Homomorphic NN Layer (per-channel)
// ============================================================
Ciphertext<DCRTPoly> HomomorphicNNForwardChannel(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input,
    const NNLayerChannel& layer,
    int numSlots)
{
    Plaintext weightPt = layer.hasCachedPlaintexts
        ? layer.weightPt
        : cc->MakeCKKSPackedPlaintext(layer.weights, 1, 0, nullptr, numSlots);
    Plaintext biasPt = layer.hasCachedPlaintexts
        ? layer.biasPt
        : cc->MakeCKKSPackedPlaintext(layer.biases, 1, 0, nullptr, numSlots);

    auto result = cc->EvalMult(input, weightPt);
    result = cc->Rescale(result);
    result = cc->EvalAdd(result, biasPt);

    if (layer.hasActivation) {
        result = cc->EvalMult(result, result);
        result = cc->Rescale(result);
    }

    return result;
}

// ============================================================
// Plaintext NN Forward Pass (per-channel)
// ============================================================
vector<double> PlaintextNNForwardChannel(
    const vector<double>& input,
    const vector<NNLayerChannel>& channelLayers)
{
    vector<double> x = input;
    for (const auto& layer : channelLayers) {
        vector<double> out(CHANNEL_SIZE);
        for (int i = 0; i < CHANNEL_SIZE; ++i) {
            out[i] = layer.weights[i] * x[i] + layer.biases[i];
            if (layer.hasActivation)
                out[i] = out[i] * out[i];
        }
        x = out;
    }
    return x;
}

// ============================================================
// Post-decryption Classification (cross-channel)
// ============================================================
int ClassifyFeatures(const vector<double>& features, const Classifier& clf) {
    vector<double> logits(clf.numClasses, 0.0);

    for (int c = 0; c < clf.numClasses; ++c) {
        for (int i = 0; i < clf.imageSize && i < (int)features.size(); ++i) {
            logits[c] += clf.weights[c][i] * features[i];
        }
        logits[c] += clf.biases[c];
    }

    // Softmax
    double maxLogit = *max_element(logits.begin(), logits.end());
    double sumExp = 0;
    vector<double> probs(clf.numClasses);
    for (int c = 0; c < clf.numClasses; ++c) {
        probs[c] = exp(logits[c] - maxLogit);
        sumExp += probs[c];
    }

    cout << "  Logits: [";
    for (int c = 0; c < clf.numClasses; ++c) {
        cout << fixed << setprecision(4) << logits[c];
        if (c < clf.numClasses - 1) cout << ", ";
    }
    cout << "]" << endl;

    cout << "  Probabilities: [";
    for (int c = 0; c < clf.numClasses; ++c) {
        probs[c] /= sumExp;
        cout << fixed << setprecision(4) << probs[c];
        if (c < clf.numClasses - 1) cout << ", ";
    }
    cout << "]" << endl;

    return max_element(logits.begin(), logits.end()) - logits.begin();
}

// ============================================================
// Save Image Files
// ============================================================
void SavePGM(const string& filename, const vector<double>& pixels, int w, int h) {
    ofstream file(filename);
    file << "P2\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i) {
        int val = max(0, min(255, (int)(pixels[i] * 255.0)));
        file << val;
        if ((i + 1) % 16 == 0) file << "\n"; else file << " ";
    }
}

void SavePPM(const string& filename, const vector<vector<double>>& channels, int w, int h) {
    if (channels.size() != 3) return;
    ofstream file(filename);
    file << "P3\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            int r = max(0, min(255, (int)(channels[0][idx] * 255.0)));
            int g = max(0, min(255, (int)(channels[1][idx] * 255.0)));
            int b = max(0, min(255, (int)(channels[2][idx] * 255.0)));
            file << r << " " << g << " " << b << " ";
        }
        file << "\n";
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    cout << "\n" << string(70, '*') << endl;
    cout << "  ENCRYPTED MEDICAL IMAGE CLASSIFICATION (RGB/Greyscale)" << endl;
    cout << "  CKKS FHE with Bootstrapping + Trained CryptoNets" << endl;
    cout << string(70, '*') << endl;

    auto totalStart = chrono::high_resolution_clock::now();
    RgbTimingMetrics timing;

    // ---- Find paths ----
    auto fileExists = [](const string& path) -> bool {
        ifstream f(path); return f.good();
    };
    auto parentDir = [](const string& path) -> string {
        size_t pos = path.find_last_of("/\\");
        return (pos != string::npos) ? path.substr(0, pos) : ".";
    };

    string exePath = parentDir(string(argv[0]));
    string weightsDir, imagesDir, modelInfoPath;

    vector<string> searchPaths = {
        "medmnist/weights", "../medmnist/weights", "../../medmnist/weights",
        "../../../medmnist/weights", exePath + "/../../../medmnist/weights",
        exePath + "/../../medmnist/weights",
    };

    for (const auto& p : searchPaths) {
        if (fileExists(p + "/weights_layer1.txt")) {
            weightsDir = p;
            imagesDir = parentDir(p) + "/test_images";
            modelInfoPath = p + "/model_info.txt";
            break;
        }
    }

    if (weightsDir.empty()) {
        cerr << "ERROR: Cannot find trained weights. Run train_model.py first." << endl;
        return 1;
    }

    // Load model info
    ModelInfo minfo = LoadModelInfo(modelInfoPath);
    cout << "\nDataset:    " << minfo.dataset << endl;
    cout << "Color mode: " << minfo.colorMode << " (" << minfo.numChannels << " channel"
         << (minfo.numChannels > 1 ? "s" : "") << ")" << endl;
    cout << "Classes:    " << minfo.numClasses << endl;
    cout << "Image size: " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << "x" << minfo.numChannels
         << " = " << minfo.imageSize << " values" << endl;

    if (minfo.numChannels == 3) {
        cout << "\n[RGB MODE] Each channel encrypted separately (3 ciphertexts)." << endl;
        cout << "Per-channel: " << CHANNEL_SIZE << " values per ciphertext." << endl;
    }

    const char* modeEnv = getenv("CKKS_CHANNEL_EXECUTION_MODE");
    ChannelExecutionMode channelMode = ParseChannelExecutionMode(modeEnv ? modeEnv : "auto");
    bool useOuterChannelParallel = ShouldUseOuterChannelParallel(channelMode, minfo.numChannels);
    cout << "Channel execution mode: " << ChannelExecutionModeName(channelMode)
         << " (outer parallel " << (useOuterChannelParallel ? "enabled" : "disabled") << ")"
         << endl;

    int numImages = (argc > 1) ? atoi(argv[1]) : 3;
    cout << "Will classify " << numImages << " test images.\n" << endl;

    // ---- Load weights ----
    auto wStart = chrono::high_resolution_clock::now();
    TrainedModel model = LoadTrainedModel(weightsDir, minfo);
    auto wEnd = chrono::high_resolution_clock::now();
    timing.weightLoadMs = chrono::duration<double, milli>(wEnd - wStart).count();

    // ---- Setup CKKS ----
    cout << "\n========================================" << endl;
    cout << "Setting up CKKS" << endl;
    cout << "========================================\n" << endl;

    auto ctxStart = chrono::high_resolution_clock::now();
    CCParams<CryptoContextCKKSRNS> parameters;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    uint32_t configuredRingDim = 1 << 14;
    parameters.SetRingDim(configuredRingDim);

    vector<uint32_t> levelBudget = {5, 5};
    uint32_t levelsAfterBoot = 10;
    uint32_t bootstrapCorrectionFactor = ResolveBootstrapCorrectionFactor();
    uint32_t approxModDepth = bootstrapCorrectionFactor;
    uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(approxModDepth, levelBudget, secretKeyDist);
    uint32_t depth = levelsAfterBoot + bootDepth;
    parameters.SetMultiplicativeDepth(depth);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);

    uint32_t maxSlots = configuredRingDim / 2;
    uint32_t numSlots = ResolveBootstrapSlots(static_cast<uint32_t>(CHANNEL_SIZE), maxSlots);
    bool warmStartRequested = ParseBoolEnv(getenv("CKKS_WARM_START"), false);
    bool cacheSaveEnabled = ParseBoolEnv(getenv("CKKS_CACHE_SAVE"), true);
    string cacheBaseDir = ResolveCacheBaseDir();
    KeyCachePaths cachePaths = BuildKeyCachePaths(cacheBaseDir, configuredRingDim, numSlots,
                                                  bootstrapCorrectionFactor);

    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    bool warmStartUsed = false;
    string warmStartReason;

    if (warmStartRequested) {
        warmStartUsed = LoadCryptoCache(cachePaths, cc, keyPair, warmStartReason);
        if (!warmStartUsed) {
            cout << "[WARM] Cache unavailable (" << warmStartReason
                 << "), falling back to cold key generation." << endl;
        }
    }

    if (!warmStartUsed) {
        cc = GenCryptoContext(parameters);
    }

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto ctxEnd = chrono::high_resolution_clock::now();
    timing.contextSetupMs = chrono::duration<double, milli>(ctxEnd - ctxStart).count();

    uint32_t actualMaxSlots = cc->GetRingDimension() / 2;
    if (numSlots > actualMaxSlots) {
        throw runtime_error("Configured bootstrap slot count exceeds loaded context capacity");
    }

    cout << "Ring dim: " << cc->GetRingDimension() << ", Max Slots: " << actualMaxSlots << endl;
    cout << "Bootstrap slots: " << numSlots << " (channel size " << CHANNEL_SIZE << ")" << endl;
    cout << "Bootstrap correction factor: " << bootstrapCorrectionFactor << endl;
    cout << "Mult depth: " << depth << ", Boot depth: " << bootDepth << endl;
    cout << "Warm-start requested: " << (warmStartRequested ? "yes" : "no") << endl;
    cout << "Warm-start used: " << (warmStartUsed ? "yes" : "no") << endl;
    if (warmStartRequested || cacheSaveEnabled) {
        cout << "Key cache profile dir: " << cachePaths.profileDir << endl;
    }

    // Bootstrap setup
    auto bsSetupStart = chrono::high_resolution_clock::now();
    cc->EvalBootstrapSetup(levelBudget, {0, 0}, numSlots, bootstrapCorrectionFactor);
    auto bsSetupEnd = chrono::high_resolution_clock::now();
    timing.bootstrapSetupMs = chrono::duration<double, milli>(bsSetupEnd - bsSetupStart).count();

    if (warmStartUsed) {
        timing.keyGenMs = 0.0;
        timing.multKeyGenMs = 0.0;
        timing.bootstrapKeyGenMs = 0.0;
        cout << "[WARM] Reused cached key material.\n" << endl;
    } else {
        auto keyStart = chrono::high_resolution_clock::now();
        keyPair = cc->KeyGen();
        auto keyEnd = chrono::high_resolution_clock::now();
        timing.keyGenMs = chrono::duration<double, milli>(keyEnd - keyStart).count();

        auto multKeyStart = chrono::high_resolution_clock::now();
        cc->EvalMultKeyGen(keyPair.secretKey);
        auto multKeyEnd = chrono::high_resolution_clock::now();
        timing.multKeyGenMs = chrono::duration<double, milli>(multKeyEnd - multKeyStart).count();

        cout << "Generating bootstrapping keys..." << endl;
        auto bootKeyStart = chrono::high_resolution_clock::now();
        cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
        auto bootKeyEnd = chrono::high_resolution_clock::now();
        timing.bootstrapKeyGenMs = chrono::duration<double, milli>(bootKeyEnd - bootKeyStart).count();
        cout << "[OK] All keys ready." << endl;

        if (cacheSaveEnabled) {
            if (SaveCryptoCache(cachePaths, cc, keyPair)) {
                cout << "[CACHE] Saved context and keys for warm-start reuse." << endl;
            } else {
                cout << "[WARN] Failed to save key cache." << endl;
            }
        }
        cout << "" << endl;
    }

        auto cacheStart = chrono::high_resolution_clock::now();
        CacheModelPlaintexts(cc, model, numSlots);
        auto cacheEnd = chrono::high_resolution_clock::now();
        cout << "Cached layer plaintext encodings in "
            << chrono::duration<double, milli>(cacheEnd - cacheStart).count()
            << " ms." << endl;

    // ========================================
    // Process test images
    // ========================================
    int numCorrect = 0, numTotal = 0;

    // Dataset-level aggregates (computed on per-image channel-averaged metrics)
    int metricsImageCount = 0;
    double sumNpcrCipher = 0.0, sumSqNpcrCipher = 0.0;
    double sumNpcr8bit = 0.0, sumSqNpcr8bit = 0.0;
    double sumUaci = 0.0, sumSqUaci = 0.0;
    double sumEntropyEnc = 0.0, sumSqEntropyEnc = 0.0;
    double sumCorrH = 0.0, sumSqCorrH = 0.0;
    double sumCorrV = 0.0, sumSqCorrV = 0.0;
    double sumCorrD = 0.0, sumSqCorrD = 0.0;
    double sumPsnr = 0.0, sumSqPsnr = 0.0;
    double sumMse = 0.0, sumSqMse = 0.0;

    auto addSample = [](double value, double& sum, double& sumSq) {
        sum += value;
        sumSq += value * value;
    };

    auto safeStd = [](double sum, double sumSq, int n) {
        if (n <= 0) return 0.0;
        double mean = sum / n;
        double var = sumSq / n - mean * mean;
        if (var < 0.0) var = 0.0;
        return sqrt(var);
    };

    for (int imgIdx = 0; imgIdx < numImages; ++imgIdx) {
        auto imageStart = chrono::high_resolution_clock::now();
        bool imageProcessed = false;
        cout << "\n" << string(60, '-') << endl;
        cout << "  Test Image " << imgIdx << " / " << numImages << endl;
        cout << string(60, '-') << endl;

        // ---- Load image per-channel ----
        vector<vector<double>> channelPixels(minfo.numChannels);

        for (int ch = 0; ch < minfo.numChannels; ++ch) {
            string imgPath;
            if (minfo.numChannels > 1) {
                imgPath = imagesDir + "/test_image_" +
                    (imgIdx < 10 ? "00" : (imgIdx < 100 ? "0" : "")) +
                    to_string(imgIdx) + "_ch" + to_string(ch) + ".txt";
            } else {
                imgPath = imagesDir + "/test_image_" +
                    (imgIdx < 10 ? "00" : (imgIdx < 100 ? "0" : "")) +
                    to_string(imgIdx) + ".txt";
            }

            if (!fileExists(imgPath)) {
                cout << "  File not found: " << imgPath << " -- skipping." << endl;
                goto next_image;
            }
            channelPixels[ch] = LoadVector(imgPath);
            if ((int)channelPixels[ch].size() != CHANNEL_SIZE) {
                cerr << "  Channel " << ch << " size mismatch: " 
                     << channelPixels[ch].size() << " vs " << CHANNEL_SIZE << endl;
                goto next_image;
            }
        }

        {
            // Load label
            string lblPath = imagesDir + "/test_image_" +
                (imgIdx < 10 ? "00" : (imgIdx < 100 ? "0" : "")) +
                to_string(imgIdx) + "_label.txt";
            int trueLabel = -1;
            { ifstream f(lblPath); if (f.good()) f >> trueLabel; }

            cout << "  True label: " << trueLabel << endl;
            if (minfo.numChannels == 3) {
                for (int ch = 0; ch < 3; ++ch) {
                    double mean = accumulate(channelPixels[ch].begin(), 
                        channelPixels[ch].end(), 0.0) / CHANNEL_SIZE;
                    cout << "  Channel " << ch << " mean: " << fixed 
                         << setprecision(4) << mean << endl;
                }
            }

            // ============================================================
            // PARALLEL PER-CHANNEL PIPELINE
            // Each channel's full pipeline (encrypt → layers 1-3 → bootstrap
            // → layers 4-5 → decrypt) is independent, so we parallelize
            // across channels when OpenMP is available.
            // With 3 RGB channels, this gives up to 3× speedup.
            // ============================================================
            vector<Ciphertext<DCRTPoly>> encChannels(minfo.numChannels);
            vector<vector<double>> decChannels(minfo.numChannels);
            vector<vector<double>> rawEncSignatures(minfo.numChannels);
            vector<double> chEncryptMs(minfo.numChannels, 0.0);
            vector<double> chPreBootMs(minfo.numChannels, 0.0);
            vector<double> chBootstrapMs(minfo.numChannels, 0.0);
            vector<double> chPostBootMs(minfo.numChannels, 0.0);
            vector<double> chDecryptMs(minfo.numChannels, 0.0);

            auto parallelStart = chrono::high_resolution_clock::now();

#ifdef _OPENMP
            if (useOuterChannelParallel) {
                cout << "  Processing " << minfo.numChannels << " channel(s) with OUTER OpenMP parallelism ("
                     << omp_get_max_threads() << " threads available)..." << endl;
            } else {
                cout << "  Processing " << minfo.numChannels
                     << " channel(s) sequentially to preserve inner OpenFHE threading..." << endl;
            }
#else
            cout << "  Processing " << minfo.numChannels << " channel(s) sequentially..." << endl;
#endif

            #pragma omp parallel for schedule(static) if(useOuterChannelParallel)
            for (int ch = 0; ch < minfo.numChannels; ++ch) {
                // --- Encrypt ---
                auto tEncStart = chrono::high_resolution_clock::now();
                Plaintext pt = cc->MakeCKKSPackedPlaintext(channelPixels[ch], 1, 0, nullptr, numSlots);
                encChannels[ch] = cc->Encrypt(keyPair.publicKey, pt);
                auto tEncEnd = chrono::high_resolution_clock::now();
                chEncryptMs[ch] = chrono::duration<double, milli>(tEncEnd - tEncStart).count();

                // Save raw-ciphertext signature before any NN processing.
                rawEncSignatures[ch] = ExtractCiphertextSignature(encChannels[ch]);

                // --- Pre-bootstrap layers (1-3) ---
                auto tPreStart = chrono::high_resolution_clock::now();
                for (int l = 0; l < 3; ++l) {
                    encChannels[ch] = HomomorphicNNForwardChannel(
                        cc, encChannels[ch], model.layers[l].channels[ch], numSlots);
                }
                auto tPreEnd = chrono::high_resolution_clock::now();
                chPreBootMs[ch] = chrono::duration<double, milli>(tPreEnd - tPreStart).count();

                // --- Bootstrap ---
                auto tBootStart = chrono::high_resolution_clock::now();
                encChannels[ch] = cc->EvalBootstrap(encChannels[ch]);
                auto tBootEnd = chrono::high_resolution_clock::now();
                chBootstrapMs[ch] = chrono::duration<double, milli>(tBootEnd - tBootStart).count();

                // --- Post-bootstrap layers (4-5) ---
                auto tPostStart = chrono::high_resolution_clock::now();
                for (int l = 3; l < 5; ++l) {
                    encChannels[ch] = HomomorphicNNForwardChannel(
                        cc, encChannels[ch], model.layers[l].channels[ch], numSlots);
                }
                auto tPostEnd = chrono::high_resolution_clock::now();
                chPostBootMs[ch] = chrono::duration<double, milli>(tPostEnd - tPostStart).count();

                // --- Decrypt ---
                auto tDecStart = chrono::high_resolution_clock::now();
                Plaintext resultPt;
                cc->Decrypt(keyPair.secretKey, encChannels[ch], &resultPt);
                resultPt->SetLength(CHANNEL_SIZE);
                decChannels[ch] = resultPt->GetRealPackedValue();
                decChannels[ch].resize(CHANNEL_SIZE);
                auto tDecEnd = chrono::high_resolution_clock::now();
                chDecryptMs[ch] = chrono::duration<double, milli>(tDecEnd - tDecStart).count();

                #pragma omp critical
                {
                    cout << "    Channel " << ch << " complete." << endl;
                }
            }

            auto parallelEnd = chrono::high_resolution_clock::now();
            double parallelMs = chrono::duration_cast<chrono::milliseconds>(
                parallelEnd - parallelStart).count();
            cout << "  All channels done in " << parallelMs << " ms." << endl;
            double sumEnc = accumulate(chEncryptMs.begin(), chEncryptMs.end(), 0.0);
            double sumPre = accumulate(chPreBootMs.begin(), chPreBootMs.end(), 0.0);
            double sumBoot = accumulate(chBootstrapMs.begin(), chBootstrapMs.end(), 0.0);
            double sumPost = accumulate(chPostBootMs.begin(), chPostBootMs.end(), 0.0);
            double sumDec = accumulate(chDecryptMs.begin(), chDecryptMs.end(), 0.0);
            timing.encryptionMs += sumEnc;
            timing.preBootstrapLayersMs += sumPre;
            timing.bootstrapEvalMs += sumBoot;
            timing.postBootstrapLayersMs += sumPost;
            timing.decryptionMs += sumDec;
            timing.channelPipelineWallMs += parallelMs;

            // Concatenate channels for classification
            vector<double> allFeatures;
            for (int ch = 0; ch < minfo.numChannels; ++ch)
                allFeatures.insert(allFeatures.end(), 
                    decChannels[ch].begin(), decChannels[ch].end());

            // ---- Classify ----
            auto clsStart = chrono::high_resolution_clock::now();
            cout << "\n  Encrypted classification:" << endl;
            int encPred = ClassifyFeatures(allFeatures, model.classifier);
            cout << "  Predicted class: " << encPred << endl;
            auto clsEnd = chrono::high_resolution_clock::now();
            timing.classificationMs += chrono::duration<double, milli>(clsEnd - clsStart).count();

            // ---- Plaintext reference (parallelized per channel) ----
            auto plainStart = chrono::high_resolution_clock::now();
            vector<vector<double>> plainChannelFeatures(minfo.numChannels);
            #pragma omp parallel for schedule(static) if(minfo.numChannels > 1)
            for (int ch = 0; ch < minfo.numChannels; ++ch) {
                // Collect per-channel layer references
                vector<NNLayerChannel> chLayers;
                for (int l = 0; l < 5; ++l)
                    chLayers.push_back(model.layers[l].channels[ch]);
                plainChannelFeatures[ch] = PlaintextNNForwardChannel(channelPixels[ch], chLayers);
            }
            vector<double> plainFeatures;
            for (int ch = 0; ch < minfo.numChannels; ++ch)
                plainFeatures.insert(plainFeatures.end(), 
                    plainChannelFeatures[ch].begin(), plainChannelFeatures[ch].end());

            cout << "\n  Plaintext classification:" << endl;
            int plainPred = ClassifyFeatures(plainFeatures, model.classifier);
            auto plainEnd = chrono::high_resolution_clock::now();
            timing.plaintextReferenceMs += chrono::duration<double, milli>(plainEnd - plainStart).count();

            bool correct = (encPred == trueLabel);
            if (correct) numCorrect++;
            numTotal++;

            cout << "\n  Encrypted pred: " << encPred << ", True: " << trueLabel
                 << (correct ? " [CORRECT]" : " [INCORRECT]") << endl;

            // ---- Security Metrics (parallelized per channel) ----
            auto secStart = chrono::high_resolution_clock::now();
            cout << "\n  Computing security metrics..." << endl;

            // Per-channel metrics storage
            vector<SecurityMetrics> channelMetrics(minfo.numChannels);
            
            #pragma omp parallel for schedule(static) if(minfo.numChannels > 1)
            for (int ch = 0; ch < minfo.numChannels; ++ch) {
                // Modified pixel for NPCR/UACI
                vector<double> modPix = channelPixels[ch];
                int ctrIdx = (IMAGE_HEIGHT / 2) * IMAGE_WIDTH + (IMAGE_WIDTH / 2);
                modPix[ctrIdx] = 1.0 - modPix[ctrIdx];

                Plaintext modPt = cc->MakeCKKSPackedPlaintext(modPix, 1, 0, nullptr, numSlots);
                auto modEnc = cc->Encrypt(keyPair.publicKey, modPt);
                vector<double> modRawEncRepr = ExtractCiphertextSignature(modEnc);

                SecurityMetricsCalculator calc(IMAGE_WIDTH, IMAGE_HEIGHT, 1);
                channelMetrics[ch] = calc.ComputeAll(
                    channelPixels[ch], rawEncSignatures[ch], decChannels[ch], {}, modRawEncRepr);

                #pragma omp critical
                {
                    string chName = (minfo.numChannels == 3) ? 
                        string(" (ch") + to_string(ch) + "=" + "RGB"[ch] + ")" : "";
                    cout << "    Channel" << chName << ": NPCR=" << fixed << setprecision(2) 
                         << channelMetrics[ch].npcr << "%, UACI=" << channelMetrics[ch].uaci 
                         << "%, PSNR=" << channelMetrics[ch].psnr 
                         << "dB, SSIM=" << setprecision(6) << channelMetrics[ch].ssim << endl;
                }
            }

            // Aggregate and print
            double avgNpcr = 0, avgUaci = 0, avgPsnr = 0, avgSsim = 0, avgEntropy = 0;
            for (int ch = 0; ch < minfo.numChannels; ++ch) {
                avgNpcr += channelMetrics[ch].npcr;
                avgUaci += channelMetrics[ch].uaci;
                avgPsnr += channelMetrics[ch].psnr;
                avgSsim += channelMetrics[ch].ssim;
                avgEntropy += channelMetrics[ch].entropyEncrypted;
            }

            // Print full report for first image, first channel
            if (imgIdx == 0) {
                SecurityMetricsCalculator::PrintSecurityReport(channelMetrics[0]);
            }

            if (minfo.numChannels > 1) {
                cout << "    Average: NPCR=" << fixed << setprecision(2)
                     << avgNpcr / minfo.numChannels << "%, UACI="
                     << avgUaci / minfo.numChannels << "%, PSNR="
                     << avgPsnr / minfo.numChannels << "dB, SSIM="
                     << setprecision(6) << avgSsim / minfo.numChannels << endl;
            }

            // Update dataset-level aggregates for requested security metrics.
            double imgNpcrCipher = 0.0;
            double imgNpcr8bit = 0.0;
            double imgUaci = 0.0;
            double imgEntropyEnc = 0.0;
            double imgCorrH = 0.0;
            double imgCorrV = 0.0;
            double imgCorrD = 0.0;
            double imgPsnr = 0.0;
            double imgMse = 0.0;

            for (int ch = 0; ch < minfo.numChannels; ++ch) {
                const auto& cm = channelMetrics[ch];
                imgNpcrCipher += cm.hasNpcrCipherTightThreshold ? cm.npcrCipherTightThreshold : cm.npcr;
                imgNpcr8bit += cm.hasNpcr8bitThreshold ? cm.npcr8bitThreshold : cm.npcr;
                imgUaci += cm.uaci;
                imgEntropyEnc += cm.entropyEncrypted;
                imgCorrH += cm.corrEncryptedH;
                imgCorrV += cm.corrEncryptedV;
                imgCorrD += cm.corrEncryptedD;
                imgPsnr += cm.psnr;
                imgMse += cm.mse;
            }

            imgNpcrCipher /= minfo.numChannels;
            imgNpcr8bit /= minfo.numChannels;
            imgUaci /= minfo.numChannels;
            imgEntropyEnc /= minfo.numChannels;
            imgCorrH /= minfo.numChannels;
            imgCorrV /= minfo.numChannels;
            imgCorrD /= minfo.numChannels;
            imgPsnr /= minfo.numChannels;
            imgMse /= minfo.numChannels;

            addSample(imgNpcrCipher, sumNpcrCipher, sumSqNpcrCipher);
            addSample(imgNpcr8bit, sumNpcr8bit, sumSqNpcr8bit);
            addSample(imgUaci, sumUaci, sumSqUaci);
            addSample(imgEntropyEnc, sumEntropyEnc, sumSqEntropyEnc);
            addSample(imgCorrH, sumCorrH, sumSqCorrH);
            addSample(imgCorrV, sumCorrV, sumSqCorrV);
            addSample(imgCorrD, sumCorrD, sumSqCorrD);
            addSample(imgPsnr, sumPsnr, sumSqPsnr);
            addSample(imgMse, sumMse, sumSqMse);
            metricsImageCount++;

            auto secEnd = chrono::high_resolution_clock::now();
            timing.securityMetricsMs += chrono::duration<double, milli>(secEnd - secStart).count();

            // Save output image
            if (minfo.numChannels == 3) {
                SavePPM("rgb_output_" + to_string(imgIdx) + ".ppm", 
                       decChannels, IMAGE_WIDTH, IMAGE_HEIGHT);
            } else {
                SavePGM("grey_output_" + to_string(imgIdx) + ".pgm",
                       decChannels[0], IMAGE_WIDTH, IMAGE_HEIGHT);
            }

            imageProcessed = true;
        }
        next_image:;
        if (imageProcessed) {
            auto imageEnd = chrono::high_resolution_clock::now();
            double imageMs = chrono::duration<double, milli>(imageEnd - imageStart).count();
            cout << "  Image total time: " << fixed << setprecision(2) << imageMs << " ms" << endl;
        }
    }

    auto totalEnd = chrono::high_resolution_clock::now();
    double totalMs = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart).count();
    timing.totalMs = totalMs;

    PrintTimingReport(timing);

        if (metricsImageCount > 0) {
           cout << "\n" << string(70, '=') << endl;
           cout << "      DATASET SECURITY METRICS SUMMARY (PER-IMAGE AVERAGES)" << endl;
           cout << string(70, '=') << endl;
           cout << "Images evaluated: " << metricsImageCount << endl;
           cout << "\nMetric                          Mean        StdDev" << endl;
           cout << string(70, '-') << endl;
           cout << fixed << setprecision(4);
           cout << "NPCR (cipher, thr=1/65535)   " << setw(9) << (sumNpcrCipher / metricsImageCount)
               << "    " << setw(9) << safeStd(sumNpcrCipher, sumSqNpcrCipher, metricsImageCount) << endl;
           cout << "NPCR (8-bit, thr=1/255)      " << setw(9) << (sumNpcr8bit / metricsImageCount)
               << "    " << setw(9) << safeStd(sumNpcr8bit, sumSqNpcr8bit, metricsImageCount) << endl;
           cout << "UACI (%)                      " << setw(9) << (sumUaci / metricsImageCount)
               << "    " << setw(9) << safeStd(sumUaci, sumSqUaci, metricsImageCount) << endl;
           cout << "Entropy (encrypted)           " << setw(9) << (sumEntropyEnc / metricsImageCount)
               << "    " << setw(9) << safeStd(sumEntropyEnc, sumSqEntropyEnc, metricsImageCount) << endl;
           cout << "Corr-H (encrypted)            " << setw(9) << (sumCorrH / metricsImageCount)
               << "    " << setw(9) << safeStd(sumCorrH, sumSqCorrH, metricsImageCount) << endl;
           cout << "Corr-V (encrypted)            " << setw(9) << (sumCorrV / metricsImageCount)
               << "    " << setw(9) << safeStd(sumCorrV, sumSqCorrV, metricsImageCount) << endl;
           cout << "Corr-D (encrypted)            " << setw(9) << (sumCorrD / metricsImageCount)
               << "    " << setw(9) << safeStd(sumCorrD, sumSqCorrD, metricsImageCount) << endl;
           cout << "PSNR (dB)                     " << setw(9) << (sumPsnr / metricsImageCount)
               << "    " << setw(9) << safeStd(sumPsnr, sumSqPsnr, metricsImageCount) << endl;
           cout << "MSE                           " << setw(9) << (sumMse / metricsImageCount)
               << "    " << setw(9) << safeStd(sumMse, sumSqMse, metricsImageCount) << endl;
           cout << string(70, '=') << "\n" << endl;
        }

    // ---- Summary ----
    cout << "\n" << string(70, '*') << endl;
    cout << "  CLASSIFICATION COMPLETE (" << minfo.colorMode << ")" << endl;
    cout << string(70, '*') << endl;
    cout << "  Dataset:     " << minfo.dataset << endl;
    cout << "  Color mode:  " << minfo.colorMode << " (" << minfo.numChannels << " channels)" << endl;
    cout << "  Accuracy:    " << numCorrect << "/" << numTotal << " ("
         << fixed << setprecision(1) 
         << (numTotal > 0 ? 100.0 * numCorrect / numTotal : 0) << "%)" << endl;
    cout << "  Total time:  " << totalMs << " ms (" << totalMs / 1000.0 << " s)" << endl;
    if (minfo.numChannels == 3) {
        cout << "  RGB: 3 ciphertexts per image, 3 bootstraps per image" << endl;
#ifdef _OPENMP
        cout << "  Parallel: OpenMP enabled (" << omp_get_max_threads() << " threads)" << endl;
#else
        cout << "  Parallel: NOT enabled (compile with OpenMP for ~3x speedup)" << endl;
#endif
    }
    cout << string(70, '*') << "\n" << endl;

    return 0;
}
