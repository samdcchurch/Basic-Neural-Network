#include <iostream>

#include "AI.hpp"
#include "Digit.hpp"
#include "PNGLayerFormatter.hpp"

int main() {
    // Query for image path
    std::cout << "Paste image path:" << std::endl;
    std::string img_path;
    std::getline(std::cin, img_path);

    // Instantiate AI
    AI Alice{AI_VARIABLES_PATH};

    // Run
    Digit guess{Alice.execute(PNGLayerFormatter::formatImage<784>(img_path.c_str()))};
    std::cout << "AI guesses: '" << guess.getValue() << "'." << std::endl;
}
