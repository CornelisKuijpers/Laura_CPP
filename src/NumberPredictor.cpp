#include "../headers/NumberPredictor.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>

NumberPredictor::NumberPredictor() :
    layer1(std::make_unique<NeuralLayer>(156, 208, 0.005)),
    layer2(std::make_unique<NeuralLayer>(208, 104, 0.005)),
    layer3(std::make_unique<NeuralLayer>(104, 52, 0.005)),
    frequencies(52, 0),
    recent_patterns(52, 0),
    position_freq(52, 0),
    total_numbers(0) {

    layer1->setMomentum(0.95);
    layer2->setMomentum(0.95);
    layer3->setMomentum(0.95);

    srand(time(nullptr));
}

void NumberPredictor::addSequence(const std::vector<int>& numbers) {
    // Decay recent patterns
    for(int i = 0; i < 52; i++) {
        recent_patterns[i] *= 0.7;  // 30% decay
    }

    // Update sequence stats
    for(size_t pos = 0; pos < numbers.size(); pos++) {
        int num = numbers[pos];
        if(num >= 1 && num <= 52) {
            frequencies[num-1]++;
            recent_patterns[num-1] = 1.0;  // Full weight for recent numbers
            position_freq[num-1]++;
            total_numbers++;
        }
    }

    sequences[sequences.size()] = numbers;
    trainOnSequence(numbers);
}

std::vector<double> NumberPredictor::predict() {
    // Create input matrix for neural network
    Matrix input(1, 156);

    // Basic frequencies
    for(int i = 0; i < 52; i++) {
        input.set(0, i, frequencies[i] / (double)total_numbers);
    }

    // Recent patterns
    for(int i = 0; i < 52; i++) {
        input.set(0, i + 52, recent_patterns[i]);
    }

    // Position frequencies
    for(int i = 0; i < 52; i++) {
        input.set(0, i + 104, position_freq[i] / (double)total_numbers);
    }

    // Get base probabilities from neural network
    std::vector<double> base_probs = getBaseProbabilities(input);

    // Adjust based on historical patterns
    std::vector<double> adjusted_probs(52, 0.0);
    for(int i = 0; i < 52; i++) {
        // Neural network prediction (60% weight)
        adjusted_probs[i] = base_probs[i] * 0.6;

        // Recent history weight (20% weight)
        adjusted_probs[i] += recent_patterns[i] * 0.2;

        // Position preference (20% weight)
        double pos_weight = position_freq[i] / (double)total_numbers;
        adjusted_probs[i] += pos_weight * 0.2;

        // Small random factor to break ties (1% weight)
        adjusted_probs[i] += (rand() / (double)RAND_MAX) * 0.01;
    }

    // Normalize final probabilities
    normalizeVector(adjusted_probs);

    return adjusted_probs;
}

std::vector<NumberProbability> NumberPredictor::predictTop7() {
    std::vector<double> all_probs = predict();

    std::vector<NumberProbability> number_probs;
    for(int i = 0; i < all_probs.size(); i++) {
        number_probs.push_back({i + 1, all_probs[i]});
    }

    std::sort(number_probs.begin(), number_probs.end());

    return std::vector<NumberProbability>(
        number_probs.begin(),
        number_probs.begin() + std::min(7ul, number_probs.size())
    );
}

std::vector<NumberCombination> NumberPredictor::predictCombinations(int numCombinations) {
    auto probabilities = predict();
    std::vector<NumberProbability> number_probs;

    // Convert to sortable format
    for(int i = 0; i < probabilities.size(); i++) {
        number_probs.push_back({i + 1, probabilities[i]});
    }

    // Sort by probability
    std::sort(number_probs.begin(), number_probs.end());

    // Generate combinations
    std::vector<NumberCombination> combinations;

    // Take top 15 most likely numbers
    std::vector<int> top_numbers;
    for(int i = 0; i < 15 && i < number_probs.size(); i++) {
        top_numbers.push_back(number_probs[i].number);
    }

    // Random device for better randomization
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate combinations
    for(int i = 0; i < numCombinations; i++) {
        NumberCombination comb;
        std::vector<int> temp = top_numbers;

        // Select EXACTLY 6 numbers
        while(comb.numbers.size() < 6 && !temp.empty()) {
            std::uniform_int_distribution<> dis(0, temp.size() - 1);
            int idx = dis(gen);
            comb.numbers.push_back(temp[idx]);
            temp.erase(temp.begin() + idx);
        }

        // Sort numbers in combination
        std::sort(comb.numbers.begin(), comb.numbers.end());

        // Only add if we got exactly 6 numbers
        if(comb.numbers.size() == 6) {
            // Calculate combination probability
            double prob = 1.0;
            for(int num : comb.numbers) {
                prob *= probabilities[num - 1];
            }
            comb.probability = prob;
            combinations.push_back(comb);
        }
    }

    // Sort combinations by probability
    std::sort(combinations.begin(), combinations.end());

    return combinations;
}

void NumberPredictor::displayCurrentData() {
    std::cout << "\nCurrent Data Analysis:\n";
    std::cout << "----------------------\n";

    std::cout << "Number frequencies:\n";
    for(int i = 0; i < frequencies.size(); i++) {
        if(frequencies[i] > 0) {
            std::cout << "Number " << (i+1) << ": " << frequencies[i]
                     << " times ("
                     << (frequencies[i] * 100.0 / total_numbers)
                     << "%)\n";
        }
    }

    std::cout << "\nStored sequences:\n";
    for(const auto& seq : sequences) {
        std::cout << "Sequence " << seq.first << ": ";
        for(int num : seq.second) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nCommon patterns:\n";
    analyzePairs();
}

void NumberPredictor::analyzePairs() {
    std::map<std::pair<int, int>, int> pair_frequencies;

    for(const auto& seq : sequences) {
        for(size_t i = 0; i < seq.second.size() - 1; i++) {
            pair_frequencies[{seq.second[i], seq.second[i+1]}]++;
        }
    }

    std::cout << "Most common consecutive pairs:\n";
    for(const auto& pair : pair_frequencies) {
        if(pair.second > 1) {
            std::cout << pair.first.first << "->" << pair.first.second
                     << ": " << pair.second << " times\n";
        }
    }
}

void NumberPredictor::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if(file.is_open()) {
        // Save frequencies
        file.write(reinterpret_cast<const char*>(frequencies.data()), frequencies.size() * sizeof(int));
        file.write(reinterpret_cast<const char*>(&total_numbers), sizeof(int));

        // Save neural networks
        layer1->save(file);
        layer2->save(file);
        layer3->save(file);

        std::cout << "Predictor saved to " << filename << std::endl;
    }
}

std::unique_ptr<NumberPredictor> NumberPredictor::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    auto predictor = std::make_unique<NumberPredictor>();

    // Load frequencies
    file.read(reinterpret_cast<char*>(predictor->frequencies.data()),
              predictor->frequencies.size() * sizeof(int));
    file.read(reinterpret_cast<char*>(&predictor->total_numbers), sizeof(int));

    // Load neural networks
    predictor->layer1 = NeuralLayer::load(file);
    predictor->layer2 = NeuralLayer::load(file);
    predictor->layer3 = NeuralLayer::load(file);

    return predictor;
}

void NumberPredictor::trainOnSequence(const std::vector<int> &sequence) {
    // Create enhanced input matrix
    Matrix input(1, 156);  // 52*3 features

    // Basic frequencies
    for(int i = 0; i < 52; i++) {
        input.set(0, i, frequencies[i] / (double)total_numbers);
    }

    // Recent appearance patterns (last 3 draws)
    for(int i = 0; i < 52; i++) {
        input.set(0, i + 52, recent_patterns[i]);
    }

    // Position frequencies
    for(int i = 0; i < 52; i++) {
        input.set(0, i + 104, position_freq[i] / (double)total_numbers);
    }

    Matrix target(1, 52, 0.0);
    for (int num : sequence) {
        if (num >= 1 && num <= 52) {
            target.set(0, num - 1, 1.0);
        }
    }

    double target_loss = 0.00001;
    double dropout_rate = 0.15;
    std::vector<double> loss_history;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < 10000; epoch++) {
        // Forward pass with dropout
        Matrix hidden1 = layer1->forward(input);
        Matrix dropout_mask1 = Matrix::createDropoutMask(hidden1.getRows(), hidden1.getCols(), dropout_rate);
        hidden1 = hidden1.hadamard(dropout_mask1);

        Matrix hidden2 = layer2->forward(hidden1);
        Matrix dropout_mask2 = Matrix::createDropoutMask(hidden2.getRows(), hidden2.getCols(), dropout_rate);
        hidden2 = hidden2.hadamard(dropout_mask2);

        Matrix output = layer3->forward(hidden2);

        double loss = Matrix::mse_loss(output, target);
        loss_history.push_back(loss);

        int correct_predictions = 0;
        for (size_t i = 0; i < 52; i++) {
            double predicted = std::round(output.get(0, i));
            double actual = target.get(0, i);
            if (std::abs(predicted - actual) < 0.1) {
                correct_predictions++;
            }
        }

        double accuracy = correct_predictions / 52.0;

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

        // Backward pass with the dropout masks
        Matrix output_gradient = output.subtract(target);
        Matrix hidden2_gradient = layer3->backward(output_gradient);
        hidden2_gradient = hidden2_gradient.hadamard(dropout_mask2);  // Apply dropout mask

        Matrix hidden1_gradient = layer2->backward(hidden2_gradient);
        hidden1_gradient = hidden1_gradient.hadamard(dropout_mask1);  // Apply dropout mask

        layer1->backward(hidden1_gradient);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nTraining Summary: \n";
    std::cout << "--------------------\n";
    std::cout << "Total Training Time: " << total_duration.count() << "ms\n";
    std::cout << "Initial Loss: " << loss_history.front() << "\n";
    std::cout << "Final Loss: " << loss_history.back() << "\n";
    std::cout << "Loss Improvement: " << (loss_history.front() - loss_history.back()) << "\n";
}

void NumberPredictor::testAccuracy(const std::vector<std::vector<int>>& test_sequences) {
    std::cout << "\nTesting Prediction Accuracy:\n";
    std::cout << "-------------------------\n";

    int total_matches = 0;
    int total_numbers = 0;

    for(const auto& actual_sequence : test_sequences) {
        // Get predictions
        auto predictions = predictTop7();
        std::set<int> predicted_numbers;
        for(const auto& pred : predictions) {
            predicted_numbers.insert(pred.number);
        }

        // Compare with actual numbers
        int matches = 0;
        std::cout << "\nActual numbers: ";
        for(int num : actual_sequence) {
            std::cout << num << " ";
            if(predicted_numbers.find(num) != predicted_numbers.end()) {
                matches++;
            }
        }

        // Show predictions
        std::cout << "\nPredicted numbers: ";
        for(const auto& pred : predictions) {
            std::cout << pred.number << "(" << (pred.probability * 100) << "%) ";
        }

        std::cout << "\nMatched " << matches << " out of " << actual_sequence.size() - 1 << " numbers\n";

        total_matches += matches;
        total_numbers += actual_sequence.size() - 1; // Excluding bonus number
    }

    double accuracy = (double)total_matches / total_numbers;
    std::cout << "\nOverall Accuracy: " << (accuracy * 100) << "%\n";
    std::cout << "Total matches: " << total_matches << " out of " << total_numbers << "\n";
}

void NumberPredictor::validateAgainstHistory() {
    std::cout << "\nValidating Against Historical Data:\n";
    std::cout << "--------------------------------\n";

    int total_hits = 0;
    int total_predictions = 0;

    // Look at last 5 sequences
    int count = 0;
    for(auto it = sequences.rbegin(); it != sequences.rend() && count < 5; ++it, ++count) {
        const auto& sequence = it->second;
        auto predictions = predictTop7();

        std::cout << "\nDraw " << it->first << ":\n";
        std::cout << "Actual:     ";
        for(int num : sequence) std::cout << num << " ";

        std::cout << "\nPredicted:  ";
        for(const auto& pred : predictions) std::cout << pred.number << " ";

        // Count matches
        int hits = 0;
        for(int num : sequence) {
            if(num == sequence.back()) continue; // Skip bonus number
            for(const auto& pred : predictions) {
                if(pred.number == num) {
                    hits++;
                    break;
                }
            }
        }

        std::cout << "\nMatched " << hits << " numbers\n";
        total_hits += hits;
        total_predictions += sequence.size() - 1; // Excluding bonus number
    }

    double accuracy = (double)total_hits / total_predictions * 100;
    std::cout << "\nOverall Historical Accuracy: " << accuracy << "%\n";
}

void NumberPredictor::validateLastPrediction(const std::vector<int>& actual_sequence) {
    std::cout << "\nValidating Last Prediction:\n";
    std::cout << "-------------------------\n";

    // Get current predictions
    auto top_predictions = predictTop7();
    auto combinations = predictCombinations(3);

    // Show actual sequence
    std::cout << "Actual sequence:    ";
    for(int num : actual_sequence) {
        std::cout << num << " ";
    }
    std::cout << "\n";

    // Show individual number predictions
    std::cout << "\nTop predicted numbers:\n";
    int matches = 0;
    for(const auto& pred : top_predictions) {
        bool was_drawn = std::find(actual_sequence.begin(),
                                 actual_sequence.end(),
                                 pred.number) != actual_sequence.end();
        std::cout << pred.number << " (" << (pred.probability * 100) << "%) "
                  << (was_drawn ? "✓" : "✗") << "\n";
        if(was_drawn) matches++;
    }

    std::cout << "Matched " << matches << " out of " << actual_sequence.size() - 1
              << " numbers (excluding bonus)\n";

    // Show combination accuracy
    std::cout << "\nPredicted combinations:\n";
    for(const auto& comb : combinations) {
        int comb_matches = 0;
        std::cout << "Combination: ";
        for(int num : comb.numbers) {
            bool was_drawn = std::find(actual_sequence.begin(),
                                     actual_sequence.end(),
                                     num) != actual_sequence.end();
            std::cout << num << (was_drawn ? "✓" : " ") << " ";
            if(was_drawn) comb_matches++;
        }
        std::cout << "\nMatched " << comb_matches << " numbers\n";
    }
}

void NumberPredictor::showPredictionStats() {
    std::cout << "\nPrediction Statistics:\n";
    std::cout << "---------------------\n";

    auto predictions = predict();

    // Show frequency vs prediction correlation
    std::vector<std::pair<int, double>> prediction_scores;
    for(int i = 0; i < 52; i++) {
        double freq_score = frequencies[i] / (double)total_numbers;
        double pred_score = predictions[i];
        prediction_scores.push_back({i + 1, pred_score});

        std::cout << "Number " << (i + 1)
                  << " - Frequency: " << (freq_score * 100) << "%"
                  << " - Predicted: " << (pred_score * 100) << "%\n";
    }

    // Sort by prediction probability
    std::sort(prediction_scores.begin(), prediction_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "\nTop 10 predicted numbers for next draw:\n";
    for(int i = 0; i < 10; i++) {
        int number = prediction_scores[i].first;
        double prob = prediction_scores[i].second;
        int freq = frequencies[number-1];

        std::cout << number << ": " << (prob * 100) << "% chance"
                  << " (drawn " << freq << " times)\n";
    }
}