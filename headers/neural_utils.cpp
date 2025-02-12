//
// Created by hadez on 2025/02/10.
//

#include "neural_utils.h"

Matrix predict(const Matrix& input, NeuralLayer& hidden_layer, NeuralLayer& output_layer) {
    Matrix hidden = hidden_layer.forward(input);
    return output_layer.forward(hidden);
}
