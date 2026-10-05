#ifndef SECURITY_METRICS_H
#define SECURITY_METRICS_H

#include <vector>
#include <string>
#include <cmath>
#include <map>

/**
 * Image Encryption Security Metrics
 * ===================================
 * 
 * Standard metrics used in image encryption research to evaluate
 * the security and quality of encryption/decryption schemes.
 * 
 * References:
 *   - Wu, Y., Noonan, J.P. and Agaian, S. (2011) "NPCR and UACI 
 *     randomness tests for image encryption"
 *   - Shannon, C.E. (1948) "A Mathematical Theory of Communication"
 *   - Wang, Z. et al. (2004) "Image Quality Assessment: From Error 
 *     Visibility to Structural Similarity" (SSIM)
 */

// ============================================================
// Structure to hold all security metrics
// ============================================================
struct SecurityMetrics {
    // --- Diffusion Metrics ---
    double npcr = 0.0;      // Number of Pixel Change Rate (%)
    double uaci = 0.0;      // Unified Average Changing Intensity (%)
    // Explicit NPCR variants to avoid ambiguity in reporting.
    double npcr8bitThreshold = 0.0;       // threshold = 1/255
    double npcrCipherTightThreshold = 0.0; // threshold = 1/65535
    bool hasNpcr8bitThreshold = false;
    bool hasNpcrCipherTightThreshold = false;
    
    // --- Randomness ---
    double entropyOriginal   = 0.0;   // Shannon entropy of original image
    double entropyEncrypted  = 0.0;   // Shannon entropy of ciphertext representation
    double entropyDecrypted  = 0.0;   // Shannon entropy of decrypted image
    
    // --- Correlation Coefficients ---
    // Original image correlations (should be high, ~0.9+)
    double corrOriginalH = 0.0;  // Horizontal
    double corrOriginalV = 0.0;  // Vertical
    double corrOriginalD = 0.0;  // Diagonal
    
    // Encrypted representation correlations (should be low, ~0.0)
    double corrEncryptedH = 0.0;
    double corrEncryptedV = 0.0;
    double corrEncryptedD = 0.0;
    
    // --- Quality Metrics (Original vs Decrypted) ---
    double mse  = 0.0;      // Mean Squared Error
    double rmse = 0.0;      // Root Mean Square Error
    double psnr = 0.0;      // Peak Signal-to-Noise Ratio (dB)
    double ssim = 0.0;      // Structural Similarity Index
    
    // --- Additional ---
    double maxAbsError   = 0.0;   // Maximum absolute error
    double meanAbsError  = 0.0;   // Mean absolute error
    double keySensitivity = 0.0;  // Key sensitivity measure (0-100%)
    
    // Image info
    int width  = 0;
    int height = 0;
    int channels = 1;
};


// ============================================================
// Security Metrics Calculator
// ============================================================
class SecurityMetricsCalculator {
public:
    /**
     * Constructor.
     * @param width  Image width
     * @param height Image height
     * @param channels Number of channels (1=greyscale, 3=RGB)
     */
    SecurityMetricsCalculator(int width, int height, int channels = 1);
    
    /**
     * Compute all security metrics.
     * 
     * @param original        Original plaintext image pixels [0, 1]
     * @param encrypted_repr  Encrypted representation (ciphertext coefficients or noise-like data)
     * @param decrypted       Decrypted image pixels
     * @param modified_decrypted  Decrypted image from 1-pixel-modified input (for NPCR/UACI)
     * @return SecurityMetrics structure with all computed values
     */
    SecurityMetrics ComputeAll(
        const std::vector<double>& original,
        const std::vector<double>& encrypted_repr,
        const std::vector<double>& decrypted,
        const std::vector<double>& modified_decrypted = {},
        const std::vector<double>& modified_encrypted_repr = {}
    );
    
    // --- Individual metric computations ---
    
    /**
     * NPCR: Number of Pixel Change Rate
     * Measures the percentage of pixels that change when one pixel of
     * the plaintext is modified. Ideal: ~99.6% for 8-bit images.
     * 
     * NPCR = (1/N) * sum(D(i)) * 100%
     * where D(i) = 0 if C1(i)==C2(i), else 1
     */
    static double ComputeNPCR(
        const std::vector<double>& image1,
        const std::vector<double>& image2,
        double threshold = 1.0 / 255.0
    );
    
    /**
     * UACI: Unified Average Changing Intensity
     * Measures the average intensity difference between two encrypted images.
     * Ideal: ~33.46% for 8-bit images.
     * 
     * UACI = (1/N) * sum(|C1(i) - C2(i)|) / maxVal * 100%
     */
    static double ComputeUACI(
        const std::vector<double>& image1,
        const std::vector<double>& image2,
        double maxVal = 1.0
    );
    
    /**
     * Shannon Information Entropy
     * Measures randomness/unpredictability. Ideal for encrypted 8-bit image: 8.0
     * 
     * H = -sum(p(i) * log2(p(i)))
     */
    static double ComputeEntropy(const std::vector<double>& data, int numBins = 256);
    
    /**
     * Adjacent Pixel Correlation Coefficient
     * Measures linear correlation between adjacent pixels.
     * Original image: high (~0.9+), Encrypted: low (~0.0)
     * 
     * @param direction 0=horizontal, 1=vertical, 2=diagonal
     */
    double ComputeCorrelation(
        const std::vector<double>& image,
        int direction,
        int numSamples = 5000
    ) const;
    
    /**
     * MSE: Mean Squared Error
     */
    static double ComputeMSE(
        const std::vector<double>& original,
        const std::vector<double>& reconstructed
    );
    
    /**
     * RMSE: Root Mean Squared Error
     */
    static double ComputeRMSE(
        const std::vector<double>& original,
        const std::vector<double>& reconstructed
    );
    
    /**
     * PSNR: Peak Signal-to-Noise Ratio (dB)
     * Higher = better reconstruction. Typical good: > 30 dB
     * 
     * PSNR = 10 * log10(MAX² / MSE)
     */
    static double ComputePSNR(
        const std::vector<double>& original,
        const std::vector<double>& reconstructed,
        double maxVal = 1.0
    );
    
    /**
     * SSIM: Structural Similarity Index Measure
     * Measures perceptual similarity. Range [0, 1], ideal = 1.0
     * 
     * SSIM(x,y) = (2*μx*μy + C1)(2*σxy + C2) / ((μx²+μy²+C1)(σx²+σy²+C2))
     * 
     * Computed per-window (8×8) and averaged.
     */
    double ComputeSSIM(
        const std::vector<double>& original,
        const std::vector<double>& reconstructed
    ) const;
    
    /**
     * Histogram analysis: compute histogram of pixel values
     */
    static std::vector<int> ComputeHistogram(
        const std::vector<double>& image,
        int numBins = 256
    );
    
    /**
     * Chi-square test for histogram uniformity
     * Lower = more uniform (desirable for encrypted images)
     */
    static double ComputeChiSquare(const std::vector<int>& histogram);
    
    /**
     * Print all security metrics in a formatted report
     */
    static void PrintSecurityReport(const SecurityMetrics& metrics);
    
private:
    int width_;
    int height_;
    int channels_;
    int totalPixels_;
};

#endif // SECURITY_METRICS_H
