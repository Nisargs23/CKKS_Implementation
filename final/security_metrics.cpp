#include "security_metrics.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <map>

using namespace std;

// ============================================================
// Constructor
// ============================================================
SecurityMetricsCalculator::SecurityMetricsCalculator(int width, int height, int channels)
    : width_(width), height_(height), channels_(channels),
      totalPixels_(width * height * channels) {}

// ============================================================
// Compute All Metrics
// ============================================================
SecurityMetrics SecurityMetricsCalculator::ComputeAll(
    const vector<double>& original,
    const vector<double>& encrypted_repr,
    const vector<double>& decrypted,
    const vector<double>& modified_decrypted,
    const vector<double>& modified_encrypted_repr)
{
    SecurityMetrics m;
    m.width = width_;
    m.height = height_;
    m.channels = channels_;
    
    // --- Quality metrics (original vs decrypted) ---
    if (!original.empty() && !decrypted.empty()) {
        m.mse  = ComputeMSE(original, decrypted);
        m.rmse = ComputeRMSE(original, decrypted);
        m.psnr = ComputePSNR(original, decrypted);
        m.ssim = ComputeSSIM(original, decrypted);
        
        // Max and mean absolute error
        double maxErr = 0.0, sumErr = 0.0;
        size_t n = min(original.size(), decrypted.size());
        for (size_t i = 0; i < n; ++i) {
            double err = abs(original[i] - decrypted[i]);
            maxErr = max(maxErr, err);
            sumErr += err;
        }
        m.maxAbsError = maxErr;
        m.meanAbsError = (n > 0) ? sumErr / n : 0.0;
    }
    
    // --- Entropy ---
    if (!original.empty())
        m.entropyOriginal = ComputeEntropy(original);
    if (!encrypted_repr.empty())
        m.entropyEncrypted = ComputeEntropy(encrypted_repr);
    if (!decrypted.empty())
        m.entropyDecrypted = ComputeEntropy(decrypted);
    
    // --- Correlation (computed on first channel for multi-channel) ---
    if (!original.empty()) {
        // Use only first channel (width_*height_ pixels) for spatial correlation
        vector<double> origChannel(original.begin(), 
            original.begin() + min((size_t)(width_ * height_), original.size()));
        m.corrOriginalH = ComputeCorrelation(origChannel, 0);
        m.corrOriginalV = ComputeCorrelation(origChannel, 1);
        m.corrOriginalD = ComputeCorrelation(origChannel, 2);
    }
    
    if (!encrypted_repr.empty()) {
        vector<double> encChannel(encrypted_repr.begin(),
            encrypted_repr.begin() + min((size_t)(width_ * height_), encrypted_repr.size()));
        m.corrEncryptedH = ComputeCorrelation(encChannel, 0);
        m.corrEncryptedV = ComputeCorrelation(encChannel, 1);
        m.corrEncryptedD = ComputeCorrelation(encChannel, 2);
    }
    
    // --- Diffusion metrics (NPCR, UACI) ---
    // Compare encrypted representations of original vs modified image
    if (!modified_encrypted_repr.empty() && !encrypted_repr.empty()) {
        // Compute both variants explicitly for transparent reporting.
        // 1/255 follows classical 8-bit NPCR-style thresholding.
        m.npcr8bitThreshold = ComputeNPCR(encrypted_repr, modified_encrypted_repr, 1.0 / 255.0);
        m.hasNpcr8bitThreshold = true;

        // Ciphertext signatures are continuous-valued; use a tighter threshold
        // than 1/255 to avoid undercounting small but real coefficient changes.
        m.npcrCipherTightThreshold = ComputeNPCR(encrypted_repr, modified_encrypted_repr, 1.0 / 65535.0);
        m.hasNpcrCipherTightThreshold = true;

        // Keep primary NPCR aligned with the ciphertext-domain evaluation.
        m.npcr = m.npcrCipherTightThreshold;
        m.uaci = ComputeUACI(encrypted_repr, modified_encrypted_repr);
    } else if (!decrypted.empty() && !modified_decrypted.empty()) {
        // For pixel-domain comparisons, only 8-bit-style threshold applies.
        m.npcr8bitThreshold = ComputeNPCR(decrypted, modified_decrypted, 1.0 / 255.0);
        m.hasNpcr8bitThreshold = true;
        m.npcr = m.npcr8bitThreshold;
        m.uaci = ComputeUACI(decrypted, modified_decrypted);
    }
    
    return m;
}

// ============================================================
// NPCR: Number of Pixel Change Rate
// ============================================================
double SecurityMetricsCalculator::ComputeNPCR(
    const vector<double>& image1,
    const vector<double>& image2,
    double threshold)
{
    if (image1.empty() || image2.empty()) return 0.0;
    size_t n = min(image1.size(), image2.size());
    
    int changed = 0;
    for (size_t i = 0; i < n; ++i) {
        if (abs(image1[i] - image2[i]) > threshold) {
            changed++;
        }
    }
    
    return 100.0 * changed / n;
}

// ============================================================
// UACI: Unified Average Changing Intensity
// ============================================================
double SecurityMetricsCalculator::ComputeUACI(
    const vector<double>& image1,
    const vector<double>& image2,
    double maxVal)
{
    if (image1.empty() || image2.empty() || maxVal <= 0) return 0.0;
    size_t n = min(image1.size(), image2.size());
    
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += abs(image1[i] - image2[i]) / maxVal;
    }
    
    return 100.0 * sum / n;
}

// ============================================================
// Shannon Information Entropy
// ============================================================
double SecurityMetricsCalculator::ComputeEntropy(const vector<double>& data, int numBins) {
    if (data.empty()) return 0.0;
    
    // Build histogram
    vector<int> hist(numBins, 0);
    for (double val : data) {
        if (!isfinite(val)) {
            continue;
        }
        // Use fractional part so values outside [0,1) still map consistently.
        double wrapped = val - floor(val);
        if (wrapped < 0.0) {
            wrapped += 1.0;
        }
        int bin = static_cast<int>(wrapped * numBins);
        bin = max(0, min(numBins - 1, bin));
        hist[bin]++;
    }
    
    // Compute entropy: H = -sum(p * log2(p))
    double entropy = 0.0;
    double n = static_cast<double>(data.size());
    
    for (int count : hist) {
        if (count > 0) {
            double p = count / n;
            entropy -= p * log2(p);
        }
    }
    
    return entropy;
}

// ============================================================
// Adjacent Pixel Correlation
// ============================================================
double SecurityMetricsCalculator::ComputeCorrelation(
    const vector<double>& image,
    int direction,
    int numSamples) const
{
    if (image.empty() || width_ <= 1 || height_ <= 1) return 0.0;
    
    // Collect pairs of adjacent pixels based on direction
    vector<pair<double, double>> pairs;
    
    mt19937 rng(42);  // Fixed seed for reproducibility
    
    // Maximum valid indices based on direction
    int maxX = width_ - 1;
    int maxY = height_ - 1;
    
    if (direction == 2) {  // diagonal
        maxX = width_ - 1;
        maxY = height_ - 1;
    }
    
    // Sample random pixel pairs
    uniform_int_distribution<int> distX(0, maxX - (direction == 0 || direction == 2 ? 1 : 0));
    uniform_int_distribution<int> distY(0, maxY - (direction == 1 || direction == 2 ? 1 : 0));
    
    for (int s = 0; s < numSamples; ++s) {
        int x = distX(rng);
        int y = distY(rng);
        int idx1 = y * width_ + x;
        
        int nx = x, ny = y;
        if (direction == 0) nx = x + 1;        // Horizontal
        else if (direction == 1) ny = y + 1;    // Vertical
        else { nx = x + 1; ny = y + 1; }       // Diagonal
        
        int idx2 = ny * width_ + nx;
        
        if (idx1 >= 0 && idx1 < (int)image.size() && 
            idx2 >= 0 && idx2 < (int)image.size()) {
            pairs.push_back({image[idx1], image[idx2]});
        }
    }
    
    if (pairs.size() < 2) return 0.0;
    
    // Compute Pearson correlation coefficient
    double sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
    double n = static_cast<double>(pairs.size());
    
    for (const auto& p : pairs) {
        sumX += p.first;
        sumY += p.second;
        sumXX += p.first * p.first;
        sumYY += p.second * p.second;
        sumXY += p.first * p.second;
    }
    
    double meanX = sumX / n;
    double meanY = sumY / n;
    
    double varX = sumXX / n - meanX * meanX;
    double varY = sumYY / n - meanY * meanY;
    double covXY = sumXY / n - meanX * meanY;
    
    double denom = sqrt(varX * varY);
    if (denom < 1e-15) return 0.0;
    
    return covXY / denom;
}

// ============================================================
// MSE: Mean Squared Error
// ============================================================
double SecurityMetricsCalculator::ComputeMSE(
    const vector<double>& original,
    const vector<double>& reconstructed)
{
    if (original.empty() || reconstructed.empty()) return 0.0;
    size_t n = min(original.size(), reconstructed.size());
    
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = original[i] - reconstructed[i];
        sum += diff * diff;
    }
    
    return sum / n;
}

// ============================================================
// RMSE: Root Mean Squared Error
// ============================================================
double SecurityMetricsCalculator::ComputeRMSE(
    const vector<double>& original,
    const vector<double>& reconstructed)
{
    return sqrt(ComputeMSE(original, reconstructed));
}

// ============================================================
// PSNR: Peak Signal-to-Noise Ratio
// ============================================================
double SecurityMetricsCalculator::ComputePSNR(
    const vector<double>& original,
    const vector<double>& reconstructed,
    double maxVal)
{
    double mse = ComputeMSE(original, reconstructed);
    if (mse < 1e-15) return 999.0;  // Practically infinite (perfect reconstruction)
    return 10.0 * log10(maxVal * maxVal / mse);
}

// ============================================================
// SSIM: Structural Similarity Index Measure
// ============================================================
double SecurityMetricsCalculator::ComputeSSIM(
    const vector<double>& original,
    const vector<double>& reconstructed) const
{
    if (original.empty() || reconstructed.empty()) return 0.0;
    
    // SSIM constants for [0, 1] range
    // C1 = (K1 * L)², C2 = (K2 * L)² where L = dynamic range = 1.0
    const double K1 = 0.01;
    const double K2 = 0.03;
    const double C1 = K1 * K1;   // 0.0001
    const double C2 = K2 * K2;   // 0.0009
    
    // Window size for local SSIM computation
    const int windowSize = 8;
    
    if (width_ < windowSize || height_ < windowSize) {
        // Image too small for windowed SSIM, compute global SSIM
        size_t n = min(original.size(), reconstructed.size());
        
        double meanX = 0, meanY = 0;
        for (size_t i = 0; i < n; ++i) {
            meanX += original[i];
            meanY += reconstructed[i];
        }
        meanX /= n;
        meanY /= n;
        
        double varX = 0, varY = 0, covXY = 0;
        for (size_t i = 0; i < n; ++i) {
            varX += (original[i] - meanX) * (original[i] - meanX);
            varY += (reconstructed[i] - meanY) * (reconstructed[i] - meanY);
            covXY += (original[i] - meanX) * (reconstructed[i] - meanY);
        }
        varX /= n;
        varY /= n;
        covXY /= n;
        
        double num = (2.0 * meanX * meanY + C1) * (2.0 * covXY + C2);
        double den = (meanX * meanX + meanY * meanY + C1) * (varX + varY + C2);
        
        return (den > 1e-15) ? num / den : 0.0;
    }
    
    // Windowed SSIM (averaged over all 8×8 windows)
    double ssimSum = 0.0;
    int windowCount = 0;
    
    // Process only the first channel for spatial SSIM
    int channelSize = width_ * height_;
    
    for (int wy = 0; wy <= height_ - windowSize; wy += windowSize / 2) {
        for (int wx = 0; wx <= width_ - windowSize; wx += windowSize / 2) {
            double sumX = 0, sumY = 0;
            double sumXX = 0, sumYY = 0, sumXY = 0;
            int count = 0;
            
            for (int dy = 0; dy < windowSize; ++dy) {
                for (int dx = 0; dx < windowSize; ++dx) {
                    int idx = (wy + dy) * width_ + (wx + dx);
                    if (idx < channelSize && idx < (int)original.size() && 
                        idx < (int)reconstructed.size()) {
                        double x = original[idx];
                        double y = reconstructed[idx];
                        sumX += x;
                        sumY += y;
                        sumXX += x * x;
                        sumYY += y * y;
                        sumXY += x * y;
                        count++;
                    }
                }
            }
            
            if (count > 1) {
                double meanX = sumX / count;
                double meanY = sumY / count;
                double varX = sumXX / count - meanX * meanX;
                double varY = sumYY / count - meanY * meanY;
                double covXY = sumXY / count - meanX * meanY;
                
                double num = (2.0 * meanX * meanY + C1) * (2.0 * covXY + C2);
                double den = (meanX * meanX + meanY * meanY + C1) * (varX + varY + C2);
                
                if (den > 1e-15) {
                    ssimSum += num / den;
                    windowCount++;
                }
            }
        }
    }
    
    return (windowCount > 0) ? ssimSum / windowCount : 0.0;
}

// ============================================================
// Histogram
// ============================================================
vector<int> SecurityMetricsCalculator::ComputeHistogram(
    const vector<double>& image,
    int numBins)
{
    vector<int> hist(numBins, 0);
    
    for (double val : image) {
        int bin = static_cast<int>(val * (numBins - 1));
        bin = max(0, min(numBins - 1, bin));
        hist[bin]++;
    }
    
    return hist;
}

// ============================================================
// Chi-Square Test for Uniformity
// ============================================================
double SecurityMetricsCalculator::ComputeChiSquare(const vector<int>& histogram) {
    if (histogram.empty()) return 0.0;
    
    double total = 0;
    for (int count : histogram) total += count;
    
    double expected = total / histogram.size();
    double chiSq = 0.0;
    
    for (int count : histogram) {
        double diff = count - expected;
        chiSq += (diff * diff) / expected;
    }
    
    return chiSq;
}

// ============================================================
// Print Security Report
// ============================================================
void SecurityMetricsCalculator::PrintSecurityReport(const SecurityMetrics& m) {
    cout << "\n" << string(70, '=') << endl;
    cout << "       IMAGE ENCRYPTION SECURITY METRICS REPORT" << endl;
    cout << string(70, '=') << endl;
    
    cout << "\n--- IMAGE INFO ---" << endl;
    cout << "Resolution:                    " << m.width << "x" << m.height;
    if (m.channels == 3)
        cout << " (RGB, " << m.channels << " channels)";
    else
        cout << " (Greyscale)";
    cout << endl;
    cout << "Total pixels:                  " << (m.width * m.height * m.channels) << endl;
    
    // --- Diffusion Metrics ---
    cout << "\n--- 1. DIFFUSION METRICS ---" << endl;
    cout << fixed << setprecision(4);
    cout << "NPCR (Number of Pixel Change Rate):" << endl;
    if (m.hasNpcrCipherTightThreshold) {
        cout << "  Measured (cipher, thr=1/65535): " << setw(10) << m.npcrCipherTightThreshold << " %" << endl;
    }
    if (m.hasNpcr8bitThreshold) {
        cout << "  Measured (8-bit style, thr=1/255): " << setw(10) << m.npcr8bitThreshold << " %" << endl;
    }
    if (!m.hasNpcrCipherTightThreshold && !m.hasNpcr8bitThreshold) {
        cout << "  Measured:                    " << setw(10) << m.npcr << " %" << endl;
    }
    cout << "  Ideal (8-bit):               " << setw(10) << "99.6094" << " %" << endl;
    if (m.npcr > 99.0) {
        cout << "  Assessment:                  EXCELLENT (high diffusion)" << endl;
    } else if (m.npcr > 90.0) {
        cout << "  Assessment:                  GOOD" << endl;
    } else if (m.npcr > 0.01) {
        cout << "  Assessment:                  MODERATE" << endl;
    } else {
        cout << "  Assessment:                  N/A (not computed)" << endl;
    }
    
    cout << "\nUACI (Unified Avg Changing Intensity):" << endl;
    cout << "  Measured:                    " << setw(10) << m.uaci << " %" << endl;
    cout << "  Ideal (8-bit):               " << setw(10) << "33.4635" << " %" << endl;
    if (m.uaci > 30.0 && m.uaci < 37.0) {
        cout << "  Assessment:                  EXCELLENT" << endl;
    } else if (m.uaci > 20.0) {
        cout << "  Assessment:                  GOOD" << endl;
    } else if (m.uaci > 0.01) {
        cout << "  Assessment:                  MODERATE" << endl;
    } else {
        cout << "  Assessment:                  N/A (not computed)" << endl;
    }
    
    // --- Entropy ---
    cout << "\n--- 2. INFORMATION ENTROPY ---" << endl;
    cout << fixed << setprecision(4);
    cout << "Entropy (Original image):      " << setw(10) << m.entropyOriginal << " bits" << endl;
    cout << "Entropy (Encrypted repr.):     " << setw(10) << m.entropyEncrypted << " bits" << endl;
    cout << "Entropy (Decrypted image):     " << setw(10) << m.entropyDecrypted << " bits" << endl;
    cout << "Ideal (8-bit):                 " << setw(10) << "8.0000" << " bits" << endl;
    if (m.entropyEncrypted > 7.9) {
        cout << "Assessment:                    EXCELLENT (near-ideal randomness)" << endl;
    } else if (m.entropyEncrypted > 7.0) {
        cout << "Assessment:                    GOOD" << endl;
    } else if (m.entropyEncrypted > 0.01) {
        cout << "Assessment:                    MODERATE" << endl;
    }
    
    // --- Correlation ---
    cout << "\n--- 3. CORRELATION COEFFICIENTS ---" << endl;
    cout << "  (Original image should be high ~0.9+; encrypted should be low ~0.0)" << endl;
    cout << fixed << setprecision(6);
    cout << "\n  Direction       Original        Encrypted       Assessment" << endl;
    cout << "  " << string(60, '-') << endl;
    
    auto corrAssess = [](double orig, double enc) -> string {
        if (abs(enc) < 0.05) return "EXCELLENT";
        if (abs(enc) < 0.1)  return "GOOD";
        if (abs(enc) < 0.3)  return "MODERATE";
        return "POOR";
    };
    
    cout << "  Horizontal   " << setw(12) << m.corrOriginalH 
         << "     " << setw(12) << m.corrEncryptedH
         << "     " << corrAssess(m.corrOriginalH, m.corrEncryptedH) << endl;
    cout << "  Vertical     " << setw(12) << m.corrOriginalV 
         << "     " << setw(12) << m.corrEncryptedV
         << "     " << corrAssess(m.corrOriginalV, m.corrEncryptedV) << endl;
    cout << "  Diagonal     " << setw(12) << m.corrOriginalD
         << "     " << setw(12) << m.corrEncryptedD
         << "     " << corrAssess(m.corrOriginalD, m.corrEncryptedD) << endl;
    
    // --- Quality Metrics ---
    cout << "\n--- 4. RECONSTRUCTION QUALITY (Original vs Decrypted) ---" << endl;
    cout << scientific << setprecision(6);
    cout << "MSE  (Mean Squared Error):     " << setw(15) << m.mse << endl;
    cout << "RMSE (Root Mean Sq Error):     " << setw(15) << m.rmse << endl;
    cout << "Max Absolute Error:            " << setw(15) << m.maxAbsError << endl;
    cout << "Mean Absolute Error:           " << setw(15) << m.meanAbsError << endl;
    cout << fixed << setprecision(4);
    cout << "PSNR (Peak SNR):               " << setw(10) << m.psnr << " dB" << endl;
    if (m.psnr > 50.0) {
        cout << "  Assessment:                  EXCELLENT (near-perfect reconstruction)" << endl;
    } else if (m.psnr > 30.0) {
        cout << "  Assessment:                  GOOD" << endl;
    } else if (m.psnr > 20.0) {
        cout << "  Assessment:                  MODERATE (noticeable quality loss)" << endl;
    } else {
        cout << "  Assessment:                  POOR" << endl;
    }
    
    // --- SSIM ---
    cout << "\n--- 5. STRUCTURAL SIMILARITY (SSIM) ---" << endl;
    cout << "SSIM:                          " << setw(10) << m.ssim << endl;
    cout << "  (1.0 = identical, 0.0 = completely different)" << endl;
    if (m.ssim > 0.99) {
        cout << "  Assessment:                  EXCELLENT" << endl;
    } else if (m.ssim > 0.95) {
        cout << "  Assessment:                  GOOD" << endl;
    } else if (m.ssim > 0.80) {
        cout << "  Assessment:                  MODERATE" << endl;
    } else {
        cout << "  Assessment:                  POOR" << endl;
    }
    
    // --- Summary ---
    cout << "\n--- SUMMARY ---" << endl;
    cout << string(70, '-') << endl;
    cout << "Metric                  Value           Ideal           Status" << endl;
    cout << string(70, '-') << endl;
    
    auto status = [](bool good) -> string { return good ? "[OK]" : "[!!]"; };
    
    cout << fixed;
    if (m.hasNpcrCipherTightThreshold) {
        cout << "NPCR (cipher)      " << setw(10) << setprecision(2) << m.npcrCipherTightThreshold
             << "       n/a            " << status(m.npcrCipherTightThreshold > 99.0 || m.npcrCipherTightThreshold < 0.01) << endl;
    }
    if (m.hasNpcr8bitThreshold) {
        cout << "NPCR (8-bit thr)   " << setw(10) << setprecision(2) << m.npcr8bitThreshold
             << "       99.61          " << status(m.npcr8bitThreshold > 99.0 || m.npcr8bitThreshold < 0.01) << endl;
    }
    if (!m.hasNpcrCipherTightThreshold && !m.hasNpcr8bitThreshold) {
        cout << "NPCR (%)           " << setw(10) << setprecision(2) << m.npcr
             << "       99.61          " << status(m.npcr > 99.0 || m.npcr < 0.01) << endl;
    }
    cout << "UACI (%)           " << setw(10) << setprecision(2) << m.uaci 
         << "       33.46          " << status((m.uaci > 30.0 && m.uaci < 37.0) || m.uaci < 0.01) << endl;
    cout << "Entropy (enc)      " << setw(10) << setprecision(4) << m.entropyEncrypted
         << "        8.0000        " << status(m.entropyEncrypted > 7.0) << endl;
    cout << "Corr-H (enc)      " << setw(10) << setprecision(6) << m.corrEncryptedH
         << "        0.0000        " << status(abs(m.corrEncryptedH) < 0.1) << endl;
    cout << "PSNR (dB)          " << setw(10) << setprecision(2) << m.psnr
         << "       >30.00         " << status(m.psnr > 30.0) << endl;
    cout << "SSIM               " << setw(10) << setprecision(6) << m.ssim
         << "        1.0000        " << status(m.ssim > 0.95) << endl;
    cout << string(70, '=') << "\n" << endl;
}
