#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <chrono>
#include <vector>
#include <fstream>

#include "headers/NeuralLayer.h"
#include "headers/neural_utils.h"
#include "headers/math/Matrix.h"
#include "headers/NumberPredictor.h"
#include "math.h"

void saveNetwork(const std::string &op, const NeuralLayer &hidden, const NeuralLayer &output) {
    std::string filename = op + "_network.bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        hidden.save(file);
        output.save(file);
        std::cout << "Network saved to " << filename << std::endl;
    } else {
        std::cout << "Error: Could not save network to " << filename << std::endl;
    }
}

std::pair<std::unique_ptr<NeuralLayer>, std::unique_ptr<NeuralLayer>> loadNetwork(const std::string &op) {
    std::string filename = op + "_network.bin";
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        auto hidden = NeuralLayer::load(file);
        auto output = NeuralLayer::load(file);
        std::cout << "Network loaded from " << filename << std::endl;
        return std::make_pair(std::move(hidden), std::move(output));
    } else {
        throw std::runtime_error("Could not load network from " + filename);
    }
}

void handleLogicGates() {
    std::vector<std::string> operations = {"AND", "OR", "NAND", "XOR"};
    std::map<std::string, std::pair<std::unique_ptr<NeuralLayer>, std::unique_ptr<NeuralLayer>>> trained_networks;

    std::cout << "Do you want to (1) Train new networks or (2) Load existing ones? ";
    int choice;
    std::cin >> choice;

    if (choice == 2) {
        try {
            for (const auto &op: operations) {
                trained_networks[op] = loadNetwork(op);
            }
            std::cout << "All networks loaded successfully!\n";
        } catch (const std::exception &e) {
            std::cout << "Error loading networks: " << e.what() << "\n";
            std::cout << "Falling back to training new networks.\n";
            choice = 1;
        }
    }

    if (choice == 1) {
        // Train all operations
        for (const auto &op: operations) {
            std::cout << "\nTraining " << op << " Gate:\n";

            double learning_rate;
            if (op == "OR") learning_rate = 0.05;
            else if (op == "AND") learning_rate = 0.2;
            else if (op == "NAND") learning_rate = 0.1;
            else if (op == "XOR") learning_rate = 0.15;

            std::unique_ptr<NeuralLayer> hidden_layer;
            std::unique_ptr<NeuralLayer> output_layer;

            if (op == "AND" || op == "NAND") {
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
                y_train.set(0, 0, 0.0); y_train.set(1, 0, 0.0);
                y_train.set(2, 0, 0.0); y_train.set(3, 0, 1.0);
            } else if (op == "OR") {
                y_train.set(0, 0, 0.0); y_train.set(1, 0, 1.0);
                y_train.set(2, 0, 1.0); y_train.set(3, 0, 1.0);
            } else if (op == "NAND") {
                y_train.set(0, 0, 1.0); y_train.set(1, 0, 1.0);
                y_train.set(2, 0, 1.0); y_train.set(3, 0, 0.0);
            } else if (op == "XOR") {
                y_train.set(0, 0, 0.0); y_train.set(1, 0, 1.0);
                y_train.set(2, 0, 1.0); y_train.set(3, 0, 0.0);
            }

            // Training loop
            double target_loss = 0.0001;
            double accuracy = 0.0;
            std::vector<double> loss_history;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int epoch = 0; epoch < 10000; epoch++) {
                Matrix hidden = hidden_layer->forward(x_train);
                Matrix output = output_layer->forward(hidden);

                double loss = Matrix::mse_loss(output, y_train);
                loss_history.push_back(loss);

                int correct_predictions = 0;
                for (size_t i = 0; i < 4; i++) {
                    double predicted = std::round(output.get(i, 0));
                    double actual = y_train.get(i, 0);
                    if (std::abs(predicted - actual) < 0.1) {
                        correct_predictions++;
                    }
                }
                accuracy = correct_predictions / 4.0;

                if (epoch % 1000 == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        current_time - start_time);
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

                Matrix output_gradient = output.subtract(y_train);
                Matrix hidden_gradient = output_layer->backward(output_gradient);
                hidden_layer->backward(hidden_gradient);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "\nTraining Summary for " << op << ":\n";
            std::cout << "------------------------\n";
            std::cout << "Total Training Time: " << total_duration.count() << "ms\n";
            std::cout << "Final Accuracy: " << (accuracy * 100) << "%\n";
            std::cout << "Initial Loss: " << loss_history.front() << "\n";
            std::cout << "Final Loss: " << loss_history.back() << "\n";
            std::cout << "Loss Improvement: " << (loss_history.front() - loss_history.back()) << "\n";

            Matrix final_hidden = hidden_layer->forward(x_train);
            Matrix final_output = output_layer->forward(final_hidden);

            std::cout << "\nTraining results for " << op << ":\n";
            std::cout << "Input 1\tInput 2\tOutput\n";
            std::cout << "------------------------\n";
            for (size_t i = 0; i < 4; i++) {
                std::cout << x_train.get(i, 0) << "\t"
                        << x_train.get(i, 1) << "\t"
                        << std::round(final_output.get(i, 0))
                        << " (raw: " << final_output.get(i, 0) << ")\n";
            }

            saveNetwork(op, *hidden_layer, *output_layer);
            trained_networks[op] = std::make_pair(std::move(hidden_layer), std::move(output_layer));
            std::cout << op << " gate trained successfully!\n";
        }
    }

    // Interactive testing loop for logic gates
    std::string selected_op;
    double input1, input2;

    while (true) {
        std::cout << "\nSelect operation (AND/OR/NAND/XOR) or 'exit' to quit: ";
        std::cin >> selected_op;

        std::transform(selected_op.begin(), selected_op.end(), selected_op.begin(), ::toupper);

        if (selected_op == "EXIT") break;

        if (trained_networks.find(selected_op) == trained_networks.end()) {
            std::cout << "Invalid operation! Please try again.\n";
            continue;
        }

        std::cout << "Enter two binary inputs (0 or 1) separated by space:\n";
        std::cin >> input1 >> input2;

        Matrix input(1, 2);
        input.set(0, 0, input1);
        input.set(0, 1, input2);

        auto &network = trained_networks[selected_op];
        Matrix hidden = network.first->forward(input);
        Matrix prediction = network.second->forward(hidden);

        std::cout << input1 << " " << selected_op << " " << input2 << " = "
                << std::round(prediction.get(0, 0))
                << " (raw output: " << prediction.get(0, 0) << ")" << std::endl;
    }
}

void handleNumberPrediction() {
    std::cout << "Number Prediction Mode:\n";
    std::cout << "1. Create new predictor\n";
    std::cout << "2. Load existing predictor\n";
    int init_choice;
    std::cin >> init_choice;

    std::unique_ptr<NumberPredictor> predictor;
    std::vector<int> most_recent_sequence;

    // Historical lottery data
    std::vector<std::vector<int>> historical_data = {
        {16, 22, 24, 34, 38, 49, 35},
        {3, 22, 23, 25, 33, 52, 19},
        {3, 7, 18, 23, 24, 52, 26},
        {3, 9, 23, 41, 42, 48, 39},
        {1, 4, 14, 17, 20, 34, 50},
        {5, 7, 18, 28, 30, 42, 1},
        {14, 15, 26, 41, 42, 52, 36},
        {6, 11, 13, 24, 32, 44, 14},
        {23, 26, 28, 29, 39, 51, 45},
        {6, 9, 23, 45, 46, 48, 12},
        {13, 20, 25, 33, 38, 40, 44},
        {4, 9, 10, 35, 37, 41, 39},
        {1, 10, 20, 22, 37, 42, 23}
    };

    if (init_choice == 2) {
        try {
            predictor = NumberPredictor::load("lottery_predictor.bin");
            std::cout << "Predictor loaded successfully!\n";
        } catch (const std::exception &e) {
            std::cout << "Error loading predictor: " << e.what() << "\n";
            std::cout << "Creating new predictor...\n";
            predictor = std::make_unique<NumberPredictor>();
        }
    } else {
        predictor = std::make_unique<NumberPredictor>();

        // Split data into training and testing sets
        size_t test_size = 3;  // Use last 3 draws for testing
        std::vector<std::vector<int>> test_data(
            historical_data.begin(),
            historical_data.begin() + test_size
        );
        std::vector<std::vector<int>> train_data(
            historical_data.begin() + test_size,
            historical_data.end()
        );

        std::cout << "Training on historical data...\n";
        for (const auto &sequence: train_data) {
            predictor->addSequence(sequence);
            std::cout << "Added sequence: ";
            for (int num: sequence) {
                std::cout << num << " ";
            }
            std::cout << std::endl;
        }

        if(!historical_data.empty()) {
            most_recent_sequence = historical_data[0];
        }

        // Test accuracy on held-out data
        std::cout << "\nTesting accuracy on recent draws:\n";
        predictor->testAccuracy(test_data);
    }

    while (true) {
        std::cout << "\nNumber Prediction Menu:\n";
        std::cout << "1. Input new sequence\n";
        std::cout << "2. Get predictions\n";
        std::cout << "3. View current data\n";
        std::cout << "4. Save predictor\n";
        std::cout << "5. Exit\n";

        int choice;
        std::cin >> choice;

        if (choice == 1) {
            std::cout << "Enter 7 numbers separated by spaces (end with -1):\n";
            std::vector<int> sequence;
            int num;
            while (std::cin >> num && num != -1 && sequence.size() < 7) {
                if (num >= 1 && num <= 52) {
                    sequence.push_back(num);
                } else {
                    std::cout << "Please enter numbers between 1 and 52\n";
                }
            }
            if (sequence.size() == 7) {
                predictor->addSequence(sequence);
                std::cout << "Sequence added successfully!\n";
                most_recent_sequence = sequence;

                // Validate predictions against this new sequence
                predictor->validateLastPrediction(sequence);
            }
        }
        else if(choice == 2) {
            // Show validation against history
            predictor->validateAgainstHistory();

            // Show prediction statistics
            predictor->showPredictionStats();

            // Show current predictions
            std::cout << "\nTop 7 most likely numbers to appear:\n";
            std::cout << "-----------------------------------\n";
            auto top_numbers = predictor->predictTop7();
            for(const auto& np : top_numbers) {
                std::cout << "Number " << np.number << ": "
                         << (np.probability * 100) << "% chance\n";
            }

            std::cout << "\nSuggested 6-Number Combinations:\n";
            std::cout << "-------------------------------\n";
            auto combinations = predictor->predictCombinations(5);
            for(size_t i = 0; i < combinations.size(); i++) {
                std::cout << "Combination " << (i+1) << ": ";
                for(int num : combinations[i].numbers) {
                    std::cout << num << " ";
                }
                std::cout << "\n(Combined probability: "
                         << (combinations[i].probability * 100) << "%)\n\n";
            }

            // Validate against most recent sequence if available
            if(!most_recent_sequence.empty()) {
                std::cout << "\nValidating against most recent draw:\n";
                predictor->validateLastPrediction(most_recent_sequence);
            }
        }
        else if (choice == 3) {
            predictor->displayCurrentData();
        }
        else if (choice == 4) {
            predictor->save("lottery_predictor.bin");
        }
        else if (choice == 5) {
            break;
        }
    }
}

int main() {
    try {
        std::cout << "Select mode:\n";
        std::cout << "1. Logic Gates\n";
        std::cout << "2. Number Prediction\n";
        int mode;
        std::cin >> mode;

        if (mode == 1) {
            handleLogicGates();
        }
        else if (mode == 2) {
            handleNumberPrediction();
        }
    }
    catch (const std::exception &e) {
        std::cout << "Error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}