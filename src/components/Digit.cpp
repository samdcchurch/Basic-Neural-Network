#include "Digit.hpp"

#include <iostream>

Digit::Digit(char c) { setValue(c); }

char Digit::getValue(void) const { return value; }

void Digit::setValue(const char c) {
    switch (c) {
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            value = c;
            break;
        default:
            std::cerr << "WARNING: attempt to set digit with value " << c
                      << ". Instead setting value to '0'" << std::endl;
            value = '0';
            break;
    }
}

bool Digit::operator==(const Digit& other) const { return value == other.value; }
