#include "AI.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

// template <size_t... LayerSizes>
// AI<LayerSizes...>::AI() : AI{-1.0, 1.0, -1.0, 0.0} {}

// template <size_t... LayerSizes>
// AI<LayerSizes...>::AI(double weight_init_lower_bound, double weight_init_upper_bound,
//                       double bias_init_lower_bound, double bias_init_upper_bound) {


// Randomize biases
// std::apply(
//     [&](auto&... arrays) {
//         ((std::for_each(arrays.begin(), arrays.end(), [&](auto& b) { b = biasDist(gen); })),
//          ...);
//     },
//     biases);
//}



// std::vector<Neuron> AI::activate(std::vector<Neuron> image) {
//     std::vector<Neuron> layer1{computeNextLayer(image, 0, 16)};
//     std::vector<Neuron> layer2{computeNextLayer(layer1, 1, 16)};
//     std::vector<Neuron> layer3{computeNextLayer(layer2, 2, 10)};
//     return layer3;
// }

// Digit AI::execute(std::vector<Neuron> image) {
//     std::vector<Neuron> result{activate(image)};
//     Neuron largest{0, 0.00};
//     for (int i{0}; i < result.size(); ++i) {
//         if (result[i].getValue() > largest.getValue()) {
//             largest = result[i];
//         }
//     }
//     return Digit{(char)(largest.getIndex() + '0')};
// }

// std::vector<double> AI::train(std::vector<Neuron> image, Digit solution) {
//     constexpr size_t l0_size{784};
//     constexpr size_t l1_size{16};
//     constexpr size_t l2_size{16};
//     constexpr size_t l3_size{10};

//     std::vector<double> gradient;
//     gradient.reserve(13002);

//     return gradient;
// }

// std::vector<Neuron> AI::computeNextLayer(std::vector<Neuron> base_layer, int base_layer_idx,
//                                          int next_layer_size) {
//     std::vector<Neuron> next_layer;
//     for (int i{0}; i < next_layer_size; ++i) {
//         double sum{0.0};
//         for (int j{0}; j < base_layer.size(); ++j) {
//             sum +=
//                 base_layer[j].getValue() *
//                 variables["weights"][base_layer_idx][j][i].get<double>();
//         }
//         sum += variables["biases"][base_layer_idx][i].get<double>();
//         double normalized_sum{sigmoid(sum)};
//         next_layer.push_back(Neuron{i, normalized_sum});
//     }
//     return next_layer;
// }

// double AI::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
