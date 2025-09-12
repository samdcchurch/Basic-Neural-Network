#include "Neuron.hpp"

#include <iostream>

Neuron::Neuron() : Neuron{0.0} {}
Neuron::Neuron(double value) { setValue(value); }

double Neuron::getValue(void) const { return value; }
void Neuron::setValue(const double d) {
    if (d < 0.00) {
        std::cerr << "WARNING: attempt to set neuron with value " << d
                  << ". Neurons only have values between 0.00 - 1.00. "
                  << "Instead setting value to 0.00." << std::endl;
        value = 0.00;
    }
    else if (d > 1.00) {
        std::cerr << "WARNING: attempt to set neuron with value " << d
                  << ". Neurons only have values between 0.00 - 1.00. "
                  << ". Instead setting value to 1.00." << std::endl;
        value = 1.00;
    }
    else {
        value = d;
    }
}
