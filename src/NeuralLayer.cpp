//
// Created by hadez on 2025/02/10.
//

#include "../headers/NeuralLayer.h"

#include <cmath>
#include <fstream>
#include <iostream>

NeuralLayer::NeuralLayer(size_t inputSize, size_t outputSize, double lr = 0.01)
    : weights(inputSize, outputSize),
      bias(1, outputSize),
      last_input(0, 0),
      last_output(0, 0),
      prev_bias_update(0,0),
      prev_weight_update(0,0),
      learning_rate(lr){
    weights.randomize(0.0, std::sqrt(2.0 / inputSize));
    bias.randomize(0.0, std::sqrt(2.0 / outputSize));
}

Matrix NeuralLayer::forward(const Matrix &input) {
    last_input = input;
    Matrix weighted_sum = input.dot(weights);

    weighted_sum = weighted_sum.add(bias);
    last_output = weighted_sum.sigmoid();
    return last_output;
}

Matrix NeuralLayer::backward(const Matrix &output_gradient) {
    // Compute activation gradient
    Matrix activation_gradient = last_output.sigmoid_derivative();
    Matrix combined_gradient = output_gradient.hadamard(activation_gradient);

    // Compute input gradient
    Matrix input_gradient = combined_gradient.dot(weights.transpose());

    // Compute weight gradients
    Matrix weight_gradient = last_input.transpose().dot(combined_gradient);

    // Compute bias gradient
    Matrix bias_gradient(1, bias.getCols());
    for(size_t j = 0; j < combined_gradient.getCols(); j++) {
        double sum = 0.0;
        for(size_t i = 0; i < combined_gradient.getRows(); i++) {
            sum += combined_gradient.get(i, j);
        }
        bias_gradient.set(0, j, sum);
    }

    // Apply momentum to weight update
    Matrix weight_update = weight_gradient.scale(-learning_rate);
    if(prev_weight_update.getRows() > 0) {  // If we have previous updates
        weight_update = weight_update.add(prev_weight_update.scale(momentum));
    }
    weights = weights.add(weight_update);
    prev_weight_update = weight_update;

    // Apply momentum to bias update
    Matrix bias_update = bias_gradient.scale(-learning_rate);
    if(prev_bias_update.getRows() > 0) {  // If we have previous updates
        bias_update = bias_update.add(prev_bias_update.scale(momentum));
    }
    bias = bias.add(bias_update);
    prev_bias_update = bias_update;

    return input_gradient;
}

void NeuralLayer::save(std::ofstream& file) const {
    // Save learning rate
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));

    // Save weights and bias
    weights.save(file);
    bias.save(file);
}

std::unique_ptr<NeuralLayer> NeuralLayer::load(std::ifstream& file) {
    double lr;

    // Load learning rate
    file.read(reinterpret_cast<char*>(&lr), sizeof(lr));

    // Load weights and bias
    Matrix weights = Matrix::load(file);
    Matrix bias = Matrix::load(file);

    // Create new layer with loaded weights and bias
    auto layer = std::make_unique<NeuralLayer>(weights.getRows(), weights.getCols(), lr);
    layer->weights = std::move(weights);
    layer->bias = std::move(bias);

    return layer;
}