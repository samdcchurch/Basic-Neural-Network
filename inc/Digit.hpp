#ifndef DIGIT_HPP
#define DIGIT_HPP

class Digit {
   public:
    explicit Digit(char);

    char getValue(void) const;
    void setValue(const char);

    bool operator==(const Digit& other) const;

   private:
    char value{'0'};
};

#endif  // DIGIT_HPP
