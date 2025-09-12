#ifndef AI_HELPERS_HPP
#define AI_HELPERS_HPP

#include <ext/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>

// Helpers for save and load
namespace {

template <typename Mat>
void writeMatrixBinary(std::ofstream& out, const Mat& mat) {
    out.write(reinterpret_cast<const char*>(mat.data()), mat.size() * sizeof(double));
}
template <typename Mat>
void readMatrixBinary(std::ifstream& in, Mat& mat) {
    in.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(double));
    if (!in) throw std::runtime_error("Failed to read matrix from file.");
}
template <typename Tuple>
void writeTupleBinary(std::ofstream& out, const Tuple& t) {
    std::apply([&](const auto&... mats) { (writeMatrixBinary(out, mats), ...); }, t);
}
template <typename Tuple>
void readTupleBinary(std::ifstream& in, Tuple& t) {
    std::apply([&](auto&... mats) { (readMatrixBinary(in, mats), ...); }, t);
}
template <size_t InputLayerSize, size_t... ProcessingLayersSizes>
void validateHeader(std::ifstream& in) {
    std::string line;
    std::getline(in, line);
    std::getline(in, line);

    const std::string prefix{"# AI Format:    AI<"};
    if (line.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error("File header unrecognized.");
    }
    std::string dimensions{line.substr(prefix.size())};
    if (dimensions.back() != '>') throw std::runtime_error("File header unrecognized.");
    dimensions.pop_back();

    std::vector<size_t> file_layers;
    while (!dimensions.empty()) {
        size_t comma{dimensions.find(',')};
        std::string dimension{dimensions.substr(0, comma)};
        file_layers.push_back(std::stoul(dimension));
        if (comma == std::string::npos) break;
        dimensions = dimensions.substr(comma + 1);
    }

    std::vector<size_t> expected_layers{InputLayerSize, ProcessingLayersSizes...};
    if (file_layers != expected_layers) {
        throw std::runtime_error("AI format does not match format in variables file.");
    }

    std::getline(in, line);
    std::getline(in, line);
    std::getline(in, line);
}

}  // anonymous namespace





namespace AIHelpers {


template <typename Tuple, typename Dist, typename Gen>
void randomizeTuple(Tuple& t, Dist& dist, Gen& gen) {
    std::apply(
        [&](auto&... mat) {
            ((mat = mat.NullaryExpr(mat.rows(), mat.cols(), [&]() { return dist(gen); })), ...);
        },
        t);
};

template <size_t InputLayerSize, size_t... ProcessingLayersSizes, typename WeightsTuple,
          typename BiasesTuple>
void save(const std::string& path, const WeightsTuple& weights, const BiasesTuple& biases) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file '" + path + "' for writing.");

    // Write header
    out << "# ===================================================" << std::endl;
    out << "# AI Format:    AI<" << InputLayerSize;
    ((out << "," << ProcessingLayersSizes), ...);
    out << ">" << std::endl;
    out << "# ===================================================" << std::endl;
    out << std::endl << std::endl;

    // Write data
    writeTupleBinary(out, weights);
    writeTupleBinary(out, biases);
}

template <size_t InputLayerSize, size_t... ProcessingLayersSizes, typename WeightsTuple,
          typename BiasesTuple>
void load(const std::string& path, WeightsTuple& weights, BiasesTuple& biases) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file '" + path + "' for reading.");

    // Validate header
    validateHeader<InputLayerSize, ProcessingLayersSizes...>(in);

    // Read data
    readTupleBinary(in, weights);
    readTupleBinary(in, biases);
}


}  // namespace AIHelpers


#endif  // AI_HELPERS_HPP
