#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <chrono>
#include <vector>

#include "headers/NeuralLayer.h"
#include "headers/neural_utils.h"
#include "headers/math/Matrix.h"
#include "math.h"

int main() {
    try {
        std::vector<std::string> operations = {"AND", "OR", "NAND", "XOR"};

        // Store trained layers for each operation using pointers
        std::map<std::string, std::pair<std::unique_ptr<NeuralLayer>, std::unique_ptr<NeuralLayer>>> trained_networks;

        // Train all operations
        for(const auto& op : operations) {
            std::cout << "\nTraining " << op << " Gate:\n";

            double learning_rate;
            if (op == "OR") {
                learning_rate = 0.05;
            } else if (op == "AND") {
                learning_rate = 0.2;
            } else if (op == "NAND") {
                learning_rate = 0.1;
            } else if (op == "XOR") {
                learning_rate = 0.15;
            }

            std::unique_ptr<NeuralLayer> hidden_layer;
            std::unique_ptr<NeuralLayer> output_layer;

            if (op == "AND") {
                hidden_layer = std::make_unique<NeuralLayer>(2, 8, 0.25);
                output_layer = std::make_unique<NeuralLayer>(8, 1, 0.15);
                hidden_layer->setMomentum(0.95);
                output_layer->setMomentum(0.95);
            } else if (op == "NAND") {  // Add these settings for NAND
                hidden_layer = std::make_unique<NeuralLayer>(2, 8, 0.25);
                output_layer = std::make_unique<NeuralLayer>(8, 1, 0.15);
                hidden_layer->setMomentum(0.95);
                output_layer->setMomentum(0.95);
            } else {
                hidden_layer = std::make_unique<NeuralLayer>(2, 5, learning_rate);
                output_layer = std::make_unique<NeuralLayer>(5, 1, learning_rate);
            }

            // Training data setup
            Matrix x_train(4, 2);
            Matrix y_train(4, 1);

            // Set inputs (same for all)
            x_train.set(0, 0, 0.0); x_train.set(0, 1, 0.0);
            x_train.set(1, 0, 0.0); x_train.set(1, 1, 1.0);
            x_train.set(2, 0, 1.0); x_train.set(2, 1, 0.0);
            x_train.set(3, 0, 1.0); x_train.set(3, 1, 1.0);

            // Set outputs based on operation
            if (op == "AND") {
                y_train.set(0, 0, 0.0); // 0 AND 0 = 0
                y_train.set(1, 0, 0.0); // 0 AND 1 = 0
                y_train.set(2, 0, 0.0); // 1 AND 0 = 0
                y_train.set(3, 0, 1.0); // 1 AND 1 = 1
            }
            else if (op == "OR") {
                y_train.set(0, 0, 0.0); // 0 OR 0 = 0
                y_train.set(1, 0, 1.0); // 0 OR 1 = 1
                y_train.set(2, 0, 1.0); // 1 OR 0 = 1
                y_train.set(3, 0, 1.0); // 1 OR 1 = 1
            }
            else if (op == "NAND") {
                y_train.set(0, 0, 1.0); // 0 NAND 0 = 1
                y_train.set(1, 0, 1.0); // 0 NAND 1 = 1
                y_train.set(2, 0, 1.0); // 1 NAND 0 = 1
                y_train.set(3, 0, 0.0); // 1 NAND 1 = 0
            }
            else if (op == "XOR") {
                y_train.set(0, 0, 0.0); // 0 XOR 0 = 0
                y_train.set(1, 0, 1.0); // 0 XOR 1 = 1
                y_train.set(2, 0, 1.0); // 1 XOR 0 = 1
                y_train.set(3, 0, 0.0); // 1 XOR 1 = 0
            }

            // Training loop
            double target_loss = 0.0001;
            double accuracy = 0.0;
            std::vector<double> loss_history;
            auto start_time = std::chrono::high_resolution_clock::now();

            for(int epoch = 0; epoch < 10000; epoch++) {
                // Forward pass
                Matrix hidden = hidden_layer->forward(x_train);
                Matrix output = output_layer->forward(hidden);

                // Calculate metrics
                double loss = Matrix::mse_loss(output, y_train);
                loss_history.push_back(loss);

                // Calculate accuracy
                int correct_predictions = 0;
                for(size_t i = 0; i < 4; i++) {
                    double predicted = std::round(output.get(i, 0));
                    double actual = y_train.get(i, 0);
                    if(std::abs(predicted - actual) < 0.1) {
                        correct_predictions++;
                    }
                }
                accuracy = correct_predictions / 4.0;

                if (epoch % 1000 == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                    std::cout << "Epoch " << epoch
                             << "\nLoss: " << loss
                             << "\nAccuracy: " << (accuracy * 100) << "%"
                             << "\nTraining Time: " << duration.count() << "ms"
                             << std::endl;
                }

                if (loss < target_loss) {
                    std::cout << "Converged at Epoch " << epoch << ", with Loss: " << loss << std::endl;
                    break;
                }

                // Backward pass
                Matrix output_gradient = output.subtract(y_train);
                Matrix hidden_gradient = output_layer->backward(output_gradient);
                hidden_layer->backward(hidden_gradient);
            }

            // After training completes
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "\nTraining Summary for " << op << ":\n";
            std::cout << "------------------------\n";
            std::cout << "Total Training Time: " << total_duration.count() << "ms\n";
            std::cout << "Final Accuracy: " << (accuracy * 100) << "%\n";
            std::cout << "Initial Loss: " << loss_history.front() << "\n";
            std::cout << "Final Loss: " << loss_history.back() << "\n";
            std::cout << "Loss Improvement: " << (loss_history.front() - loss_history.back()) << "\n";

            // Verify training results
            Matrix final_hidden = hidden_layer->forward(x_train);
            Matrix final_output = output_layer->forward(final_hidden);

            std::cout << "\nTraining results for " << op << ":\n";
            std::cout << "Input 1\tInput 2\tOutput\n";
            std::cout << "------------------------\n";
            for(size_t i = 0; i < 4; i++) {
                std::cout << x_train.get(i, 0) << "\t"
                         << x_train.get(i, 1) << "\t"
                         << std::round(final_output.get(i, 0))
                         << " (raw: " << final_output.get(i, 0) << ")\n";
            }

            // Store trained networks
            trained_networks[op] = std::make_pair(std::move(hidden_layer), std::move(output_layer));
            std::cout << op << " gate trained successfully!\n";
        }

        // Interactive testing loop
        std::string selected_op;
        double input1, input2;

        while(true) {
            std::cout << "\nSelect operation (AND/OR/NAND/XOR) or 'exit' to quit: ";
            std::cin >> selected_op;

            std::transform(selected_op.begin(), selected_op.end(), selected_op.begin(), ::toupper);

            if(selected_op == "EXIT") break;

            if(trained_networks.find(selected_op) == trained_networks.end()) {
                std::cout << "Invalid operation! Please try again.\n";
                continue;
            }

            std::cout << "Enter two binary inputs (0 or 1) separated by space:\n";
            std::cin >> input1 >> input2;

            Matrix input(1, 2);
            input.set(0, 0, input1);
            input.set(0, 1, input2);

            auto& network = trained_networks[selected_op];
            Matrix hidden = network.first->forward(input);
            Matrix prediction = network.second->forward(hidden);

            std::cout << input1 << " " << selected_op << " " << input2 << " = "
                     << std::round(prediction.get(0, 0))
                     << " (raw output: " << prediction.get(0, 0) << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}