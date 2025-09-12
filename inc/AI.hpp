#ifndef AI_HPP
#define AI_HPP

#include <algorithm>
#include <cmath>
#include <ext/json.hpp>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "AIHelpers.hpp"
#include "AITypes.hpp"

template <size_t InputLayerSize, size_t... ProcessingLayersSizes>
class AI {
   public:
    AI(std::string load_data_path) : save_data_path{load_data_path} {
        AIHelpers::load<InputLayerSize, ProcessingLayersSizes...>(load_data_path, weights, biases);
    }
    AI(std::string save_data_path, double weight_init_lower_bound, double weight_init_upper_bound,
       double bias_init_lower_bound, double bias_init_upper_bound)
        : save_data_path{save_data_path} {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> weightDist(weight_init_lower_bound,
                                                          weight_init_upper_bound);
        std::uniform_real_distribution<double> biasDist(bias_init_lower_bound,
                                                        bias_init_upper_bound);

        AIHelpers::randomizeTuple(weights, weightDist, gen);
        AIHelpers::randomizeTuple(biases, biasDist, gen);
        AIHelpers::save<InputLayerSize, ProcessingLayersSizes...>(save_data_path, weights, biases);
    }


   private:
    AITypes::Layers<InputLayerSize, ProcessingLayersSizes...> layers;
    AITypes::Weights<InputLayerSize, ProcessingLayersSizes...> weights;
    AITypes::Biases<ProcessingLayersSizes...> biases;
    std::string save_data_path;
};

// class AI {
//    public:

//     std::vector<Neuron> activate(std::vector<Neuron> image);
//     Digit execute(std::vector<Neuron> image);
//     std::vector<double> train(std::vector<Neuron> image, Digit solution);

//    private:
//     std::vector<Neuron> computeNextLayer(std::vector<Neuron> base_layer, int base_layer_idx,
//                                          int next_layer_size);
//     double sigmoid(double x);
// };

#endif  // AI_HPP
