//
// Created by hadez on 2025/02/10.
//

#ifndef NEURAL_UTILS_H
#define NEURAL_UTILS_H
#include "NeuralLayer.h"

Matrix predict(const Matrix& input, NeuralLayer& hidden_layer, NeuralLayer& output_layer);

#endif //NEURAL_UTILS_H
