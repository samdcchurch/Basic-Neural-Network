#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "AI.hpp"
#include "Digit.hpp"
#include "Neuron.hpp"
#include "PNGLayerFormatter.hpp"
#include "Sample.hpp"

template <int N>
void generateImageSeed(std::array<int, N>&);
std::string getFile(int index);

int main() {
    // Generate seed
    constexpr int NUM_IMAGES{60000};
    std::array<int, NUM_IMAGES> seed;
    generateImageSeed(seed);

    // Instantiate AI
    AI Alice{AI_VARIABLES_PATH};

    // Stochastic graident decent with mini-bathces of size 100
    constexpr size_t BATCH_SIZE{100};

    for (size_t i{0}; i < seed.size(); i += BATCH_SIZE) {
        size_t end{std::min(i + BATCH_SIZE, seed.size())};
        std::vector<std::vector<double>> gradients;
        for (size_t j{0}; j < end; ++j) {
            std::string img_path{getFile(seed[i + j])};
            gradients.push_back(Alice.train(PNGLayerFormatter::formatImage<784>(img_path.c_str()),
                                            Digit{img_path[img_path.find("__") + 2]}));
        }
        // TODO: Average all entries in gradients and edit JSON based on avg. gradient
    }
}

template <int N>
void generateImageSeed(std::array<int, N>& seed) {
    for (int i{0}; i < N; ++i) {
        seed[i] = i;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(seed.begin(), seed.end(), gen);
};

std::string getFile(int index) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << index;
    std::string prefix = oss.str();

    std::filesystem::path folder_path(AI_TRAIN_DATA_DIR);
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string filename{entry.path().filename().string()};
            if (filename.rfind(prefix, 0) == 0) {
                return entry.path().string();
            }
        }
    }
    std::cerr << "WARNING: Could not find test image with index " << index << "." << std::endl;
    return "";
};
