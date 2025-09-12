#ifndef SAMPLE_HPP
#define SAMPLE_HPP

#include <vector>

#include "Digit.hpp"
#include "Neuron.hpp"

struct Sample {
    std::vector<Neuron> image;
    Digit answer;
};

#endif  // SAMPLE_HPP