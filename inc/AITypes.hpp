#ifndef AI_TYPES_HPP
#define AI_TYPES_HPP

#include <ext/Eigen/Dense>
#include <tuple>

#include "Neuron.hpp"

template <size_t... Layers>
struct WeightTupleBuilder;


namespace AITypes {

template <size_t InputLayerSize, size_t... ProcessingLayersSizes>
using Layers = std::tuple<Eigen::Matrix<Neuron, InputLayerSize, 1>,
                          Eigen::Matrix<Neuron, ProcessingLayersSizes, 1>...>;

template <size_t InputLayerSize, size_t... ProcessingLayersSizes>
using Weights = typename WeightTupleBuilder<InputLayerSize, ProcessingLayersSizes...>::type;

template <size_t... ProcessingLayersSizes>
using Biases = std::tuple<Eigen::Matrix<double, ProcessingLayersSizes, 1>...>;

}  // namespace AITypes


namespace {


template <size_t First, size_t Second, size_t... Rest>
struct WeightTupleBuilder<First, Second, Rest...> {
    using type = decltype(std::tuple_cat(std::tuple<Eigen::Matrix<double, First, Second>>{},
                                         typename WeightTupleBuilder<Second, Rest...>::type{}));
};
template <size_t Last>
struct WeightTupleBuilder<Last> {
    using type = std::tuple<>;
};


}  // anonymous namespace


#endif  // AI_TYPES_HPP
