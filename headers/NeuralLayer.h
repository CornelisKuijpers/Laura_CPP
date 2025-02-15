//
// Created by hadez on 2025/02/10.
//

#ifndef NEURALLAYER_H
#define NEURALLAYER_H
#include <memory>

#include "math/Matrix.h"

class NeuralLayer {
public:
    NeuralLayer(size_t inputSize, size_t outputSize, double lr);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& output_gradient);
    void setMomentum(double m) { momentum = m; }
    void save(std::ofstream& file) const;
    static std::unique_ptr<NeuralLayer> load(std::ifstream& file);
private:
    Matrix weights;
    Matrix bias;
    Matrix last_input;
    Matrix last_output;
    double learning_rate;
    double momentum = 0.9;
    Matrix prev_weight_update;
    Matrix prev_bias_update;

};

#endif //NEURALLAYER_H
