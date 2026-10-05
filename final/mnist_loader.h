#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>

class MNISTLoader {
public:
    // Load MNIST images from IDX file format
    static std::vector<std::vector<double>> loadImages(const std::string& filename, int maxImages = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return {};
        }

        // Read header
        uint32_t magic = readUint32(file);
        if (magic != 2051) {
            std::cerr << "Error: Invalid MNIST image file magic number: " << magic << std::endl;
            return {};
        }

        uint32_t numImages = readUint32(file);
        uint32_t numRows = readUint32(file);
        uint32_t numCols = readUint32(file);

        std::cout << "MNIST Images: " << numImages << " images, " 
                  << numRows << "x" << numCols << " pixels each" << std::endl;

        if (maxImages > 0 && maxImages < (int)numImages) {
            numImages = maxImages;
        }

        std::vector<std::vector<double>> images(numImages);
        
        for (uint32_t i = 0; i < numImages; ++i) {
            images[i].resize(numRows * numCols);
            for (uint32_t j = 0; j < numRows * numCols; ++j) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                // Normalize to [0, 1]
                images[i][j] = static_cast<double>(pixel) / 255.0;
            }
        }

        file.close();
        return images;
    }

    // Load MNIST labels from IDX file format
    static std::vector<uint8_t> loadLabels(const std::string& filename, int maxLabels = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return {};
        }

        // Read header
        uint32_t magic = readUint32(file);
        if (magic != 2049) {
            std::cerr << "Error: Invalid MNIST label file magic number: " << magic << std::endl;
            return {};
        }

        uint32_t numLabels = readUint32(file);
        
        std::cout << "MNIST Labels: " << numLabels << " labels" << std::endl;

        if (maxLabels > 0 && maxLabels < (int)numLabels) {
            numLabels = maxLabels;
        }

        std::vector<uint8_t> labels(numLabels);
        file.read(reinterpret_cast<char*>(labels.data()), numLabels);

        file.close();
        return labels;
    }

    // Get image dimensions
    static constexpr int getImageWidth() { return 28; }
    static constexpr int getImageHeight() { return 28; }
    static constexpr int getImageSize() { return 28 * 28; }  // 784 pixels

    // Print a single image to console (for debugging)
    static void printImage(const std::vector<double>& image, int width = 28, int height = 28) {
        const char* shades = " .:-=+*#%@";
        int numShades = 10;
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int shadeIdx = static_cast<int>(image[idx] * (numShades - 1));
                shadeIdx = std::min(std::max(shadeIdx, 0), numShades - 1);
                std::cout << shades[shadeIdx] << shades[shadeIdx];
            }
            std::cout << std::endl;
        }
    }

private:
    static uint32_t readUint32(std::ifstream& file) {
        uint32_t value = 0;
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        // MNIST uses big-endian
        value = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        return value;
    }
};

#endif // MNIST_LOADER_H
