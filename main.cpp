#include <iostream>
#include "NeuralNetwork.h"

int main() {
    NeuralNetwork network(3,4);
    std::vector<double> input{0.1,0.4,0.6};
    network.run(input);
    return 0;
}