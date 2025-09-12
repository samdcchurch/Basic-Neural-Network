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
    constexpr int NUM_IMAGES{10000};
    std::array<int, NUM_IMAGES> seed;
    generateImageSeed(seed);

    // Instantiate AI
    AI Alice{AI_VARIABLES_PATH};

    double avg_cost{0.00};
    for (int number : seed) {
        std::string img_path{getFile(number)};
        std::vector<Neuron> activation{
            Alice.activate(PNGLayerFormatter::formatImage<784>(img_path.c_str()))};
        Digit solution{img_path[img_path.find("__") + 2]};

        double cost{0.0};
        for (int i{0}; i < 10; ++i) {
            if (i == solution.getValue() - '0') {
                cost += std::pow((activation[i].getValue() - 1.00), 2);
            }
            else {
                cost += std::pow((activation[i].getValue() - 0.00), 2);
            }
        }
        avg_cost += cost;
    }
    avg_cost = avg_cost / (double)NUM_IMAGES;
    std::cout << "Average cost: " << avg_cost << "." << std::endl;
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

    std::filesystem::path folder_path(AI_TEST_DATA_DIR);
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
