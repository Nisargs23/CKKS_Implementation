#include "image_processor.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>

ImageProcessor::ImageProcessor(int width, int height) 
    : width_(width), height_(height) {}

std::vector<double> ImageProcessor::generateDummyImage() {
    std::vector<double> image(width_ * height_);
    
    // Create a simple pattern: gradient with some geometric shapes
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int idx = y * width_ + x;
            
            // Base gradient
            double value = (double)(x + y) / (width_ + height_);
            
            // Add a circle in the center
            int cx = width_ / 2;
            int cy = height_ / 2;
            double dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (dist < std::min(width_, height_) / 4.0) {
                value += 0.5;
            }
            
            // Add a square in the corner
            if (x < width_ / 4 && y < height_ / 4) {
                value += 0.3;
            }
            
            // Normalize to [0, 1]
            image[idx] = std::min(1.0, value);
        }
    }
    
    return image;
}

std::vector<double> ImageProcessor::normalizeImage(const std::vector<double>& image) {
    std::vector<double> normalized = image;
    
    // Find min and max
    double minVal = *std::min_element(image.begin(), image.end());
    double maxVal = *std::max_element(image.begin(), image.end());
    
    // Normalize to [0, 1]
    if (maxVal - minVal > 1e-10) {
        for (size_t i = 0; i < normalized.size(); ++i) {
            normalized[i] = (image[i] - minVal) / (maxVal - minVal);
        }
    }
    
    return normalized;
}

std::vector<double> ImageProcessor::applyBrightnessFilter(
    const std::vector<double>& image, double factor) {
    
    std::vector<double> filtered(image.size());
    
    for (size_t i = 0; i < image.size(); ++i) {
        filtered[i] = std::min(1.0, std::max(0.0, image[i] * factor));
    }
    
    return filtered;
}

void ImageProcessor::printImageStats(const std::vector<double>& image, 
                                     const std::string& label) {
    double sum = std::accumulate(image.begin(), image.end(), 0.0);
    double mean = sum / image.size();
    
    double sq_sum = std::inner_product(image.begin(), image.end(), 
                                       image.begin(), 0.0);
    double variance = sq_sum / image.size() - mean * mean;
    double stddev = std::sqrt(variance);
    
    double minVal = *std::min_element(image.begin(), image.end());
    double maxVal = *std::max_element(image.begin(), image.end());
    
    std::cout << "\n" << label << " Statistics:" << std::endl;
    std::cout << "  Size: " << width_ << "x" << height_ << " (" << image.size() << " pixels)" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std Dev: " << stddev << std::endl;
    std::cout << "  Min: " << minVal << std::endl;
    std::cout << "  Max: " << maxVal << std::endl;
}

void ImageProcessor::saveImage(const std::vector<double>& image, 
                               const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Simple PGM format (ASCII)
    file << "P2\n";
    file << width_ << " " << height_ << "\n";
    file << "255\n";
    
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int idx = y * width_ + x;
            int pixelValue = static_cast<int>(image[idx] * 255.0);
            file << pixelValue << " ";
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Image saved to: " << filename << std::endl;
}
