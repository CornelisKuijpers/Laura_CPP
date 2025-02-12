//
// Created by hadez on 2025/02/10.
//

#include "../../headers/math/Matrix.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <random>

#include "../../headers/NeuralLayer.h"

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(size_t rows, size_t cols, double value) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, value));
}

void Matrix::randomize(double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // He initialization
    double he_stddev = std::sqrt(2.0 / rows);
    std::normal_distribution<> distrib(0, he_stddev);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            data[i][j] = distrib(gen);
        }
    }
}

Matrix Matrix::dot(const Matrix &other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix::dot - Matrix dimensions don't match for multiplication");
    }

    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix &other) const {
    // Special case for bias addition (broadcasting)
    if(other.rows == 1 && cols == other.cols) {
        Matrix result(rows, cols);
        for(size_t i = 0; i < rows; i++) {
            for(size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[0][j];
            }
        }
        return result;
    }

    if(rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix::add - Matrix dimensions don't match for addition");
    }

    Matrix result(rows, cols);
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::activate() const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i][j] = std::max(0.0, data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix::hadamard - Matrix dimensions don't match for hadamard");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::relu_derivative() const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] > 0 ? 1.0 : 0.0;
        }
    }
    return result;
}

Matrix Matrix::scale(double factor) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * factor;
        }
    }
    return result;
}

double Matrix::mse_loss(const Matrix &predictions, const Matrix &targets) {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Matrix::mse_loss - Matrix dimensions don't match for mse_loss");
    }

    double sum = 0.0;
    size_t total = predictions.rows * predictions.cols;

    for (size_t i = 0; i < predictions.rows; i++) {
        for (size_t j = 0; j < predictions.cols; j++) {
            double diff = predictions.data[i][j] - targets.data[i][j];
            sum += diff * diff;
        }
    }

    return sum/total;
}

void Matrix::set(size_t row, size_t col, double value) {
    data[row][col] = value;
}

double Matrix::get(size_t row, size_t col) const {
    return data[row][col];
}

Matrix Matrix::subtract(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix::subtract - Matrix dimensions don't match for subtract");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

void Matrix::print(const std::string &label) const {
    if (!label.empty()) {
        std::cout << label << " ";
    }
    std::cout << "(" << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << data[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

Matrix Matrix::sigmoid() const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.data[i][j] = 1.0 / (1.0 + exp(-data[i][j]));
        }
    }
    return result;
}

Matrix Matrix::sigmoid_derivative() const {
    Matrix sigmoid_output = sigmoid();
    return sigmoid_output.hadamard(
        Matrix(rows, cols, 1.0).subtract(sigmoid_output)  // 1 - sigmoid
    );
}

void Matrix::save(std::ofstream &file) const {
    // Save dimensions
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Save data
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            file.write(reinterpret_cast<const char*>(&data[i][j]), sizeof(double));
        }
    }
}

Matrix Matrix::load(std::ifstream &file) {
    size_t rows, cols;

    // Load dimensions
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    Matrix result(rows, cols);

    // Load data
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            file.read(reinterpret_cast<char*>(&result.data[i][j]), sizeof(double));
        }
    }

    return result;  // Return Matrix instead of NeuralLayer
}
