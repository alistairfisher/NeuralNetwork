//
// Created by Alistair Fisher on 03/05/2018.
//

#include <iostream>
#include <random>
#include "NeuralNetwork.h"

namespace {
    std::ostream& operator<<(std::ostream& os, const std::vector<double>& vector) {
        for (auto cit = vector.cbegin();cit!=vector.cend();++cit) {
            os << *cit;
        }
        return os;
    }
}

NeuralNetwork::NeuralNetwork(int inputs, int layers):
  weights(layers,std::vector<std::vector<double>>(inputs,std::vector<double>(inputs,1)))
{
    std::default_random_engine generator;
    std::uniform_real_distribution<> dist(0, 1);
    auto rand = std::bind(dist,generator);
    for (auto outer_it = weights.begin();outer_it != weights.end();++outer_it) {
        for (auto inner_it = outer_it->begin();inner_it != outer_it->end();inner_it++) {
            std::generate(inner_it->begin(),inner_it->end(),rand);
            std::cout << *inner_it;
        }
        std::cout << std::endl;
    }
}

void NeuralNetwork::train(std::vector<double> inputs, std::vector<double> correctOutputs) {
    std::vector<double> output = run(inputs);
    std::vector<double> error;
    std::transform(output.cbegin(),output.cend(),correctOutputs.cbegin(),std::back_inserter(error),std::minus<double>());
    std::for_each(weights.rbegin(),weights.rend(),[this,&error] (WeightsMatrix& weightsMatrix) -> void
    {this->backPropagateOneLayer(weightsMatrix,error);});
}

std::vector<double> NeuralNetwork::run(std::vector<double> inputs) {
    if (inputs.size() != expectedInputs()) {
        throw std::runtime_error("Wrong number of inputs");
    }
    auto result = inputs;
    using namespace std::placeholders;
    std::for_each(weights.cbegin(),weights.cend(),[this,&inputs](const WeightsMatrix& w)->void{
            this->applyWeightLayer(w,inputs);});
    return result;
}

int NeuralNetwork::expectedInputs() {
    return weights[0][0].size();
}

void NeuralNetwork::applyWeightLayer(const WeightsMatrix& weights, std::vector<double>& inputs) {
    for (int i = 0; i < inputs.size();++i) {
        inputs[i] = 0;
        for (int j = 0; j < inputs.size();++j) {
            inputs[i] += inputs[j] * weights[j][i];
        }
    }
}

void NeuralNetwork::backPropagateOneLayer(WeightsMatrix& weights, std::vector<double>& inputs) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = std::inner_product(weights[i].begin(),weights[i].end(),inputs.begin(),0); // Could use map_reduce
        // here
    }
}