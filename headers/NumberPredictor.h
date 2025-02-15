//
// Created by hadez on 2025/02/12.
//

#ifndef NUMBER_PREDICTOR_H
#define NUMBER_PREDICTOR_H

#include <memory>
#include <vector>
#include <map>
#include "NeuralLayer.h"

class NumberPredictor {
private:
    std::unique_ptr<NeuralLayer> hidden_layer;
    std::unique_ptr<NeuralLayer> output_layer;
    std::vector<int> frequencies;
    int total_numbers;
    std::map<int, std::vector<int>> sequences;  // Store sequences for analysis

public:
    NumberPredictor();
    void addSequence(const std::vector<int>& numbers);
    std::vector<double> predict();
    void displayCurrentData();

private:
    void analyzePairs();
};

#endif //NUMBER_PREDICTOR_H