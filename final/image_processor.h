#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <string>
#include "openfhe/pke/openfhe.h"

using namespace lbcrypto;

class ImageProcessor {
public:
    // Constructor
    ImageProcessor(int width, int height);
    
    // Generate a dummy image (simple pattern)
    std::vector<double> generateDummyImage();
    
    // Normalize image pixels to range suitable for CKKS
    std::vector<double> normalizeImage(const std::vector<double>& image);
    
    // Apply a simple filter (e.g., brightness adjustment)
    std::vector<double> applyBrightnessFilter(const std::vector<double>& image, double factor);
    
    // Print image statistics
    void printImageStats(const std::vector<double>& image, const std::string& label);
    
    // Save image to file (simple format)
    void saveImage(const std::vector<double>& image, const std::string& filename);
    
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getSize() const { return width_ * height_; }
    
private:
    int width_;
    int height_;
};

#endif // IMAGE_PROCESSOR_H
