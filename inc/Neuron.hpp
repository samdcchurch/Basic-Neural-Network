#ifndef NEURON_HPP
#define NEURON_HPP

#include <memory>

#include "Weight.hpp"

class Neuron {
   public:
    Neuron();
    Neuron(double value);

    double getValue(void) const;
    void setValue(const double);

   private:
    double value{0};
};

#endif  // NEURON_HPP
