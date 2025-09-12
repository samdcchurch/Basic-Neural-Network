#ifndef PNG_NEURON_FORMATTER_HPP
#define PNG_NEURON_FORMATTER_HPP

#define STB_IMAGE_IMPLEMENTATION
#include <ext/stb_image.h>
#include <iostream>

#include "AILayer.hpp"
#include "Neuron.hpp"

namespace PNGLayerFormatter {

template <size_t N>
AILayer<N> formatImage(const char* IMAGE_PATH) {
    int width{0}, height{0}, channels{0};
    unsigned char* data = stbi_load(IMAGE_PATH, &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "ERROR: Failed to load image: \"" << IMAGE_PATH << "\"." << std::endl;
        return AILayer<N>{};
    }
    if (width * height != N) {
        std::cerr << "ERROR: Image resolution: \"" << width << " x " << height
                  << "\" does not match input layer size of '" << N << "'." << std::endl;
        return AILayer<N>{};
    }

    AILayer<N> image;

    for (size_t i{0}; i < N; ++i) {
        double normalized_grayscale_value{data[i] / 255.0};
        image[i].setValue(normalized_grayscale_value);
    }

    stbi_image_free(data);

    return image;
}

}  // namespace PNGLayerFormatter

#endif  // PNG_NEURON_FORMATTER_HPP
