//
// Created by hadez on 2025/02/10.
//

#ifndef MATRIX_H
#define MATRIX_H
#include <memory>
#include <string>
#include <vector>
#include <fstream>  // Add this for file operations
#include <random>

class NeuralLayer;

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);
    void randomize(double mean, double stddev);
    Matrix dot(const Matrix &other) const;
    Matrix add(const Matrix &other) const;
    Matrix activate() const;
    Matrix transpose() const;
    Matrix hadamard(const Matrix &other) const;
    Matrix relu_derivative() const;
    Matrix scale(double factor) const;
    static double mse_loss(const Matrix& predictions, const Matrix& targets);
    void set(size_t row, size_t col, double value);
    double get(size_t row, size_t col) const;
    Matrix subtract(const Matrix &other) const;

    size_t getRows() const {return rows;};
    size_t getCols() const {return cols;};

    void print(const std::string& label = "") const;
    Matrix sigmoid() const;
    Matrix sigmoid_derivative() const;

    void save(std::ofstream& file) const;
    static Matrix load(std::ifstream& file);
    static Matrix createDropoutMask(size_t rows, size_t cols, double dropout_rate) {
        Matrix mask(rows, cols);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                mask.data[i][j] = (dis(gen) > dropout_rate) ? 1.0 : 0.0;
            }
        }
        return mask;
    }
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;
};

#endif //MATRIX_H