//
// Created by Alistair Fisher on 03/05/2018.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H


#include <vector>

class NeuralNetwork {

public:

    NeuralNetwork(int inputs, int layers); // For now, inputs = outputs

    void train(std::vector<double> inputs, std::vector<double> outputs);

    std::vector<double> run(std::vector<double> inputs);

    int expectedInputs();

private:

    typedef std::vector<std::vector<double>> WeightsMatrix;
    // Represents the weights at a single layer. Inner matrix represents all outgoing edges of a single node

    std::vector<WeightsMatrix> weights;

    void applyWeightLayer(const WeightsMatrix&, std::vector<double>& inputs);

    void backPropagateOneLayer(WeightsMatrix&, std::vector<double>& inputs);
};


#endif //NEURALNETWORK_NEURALNETWORK_H
