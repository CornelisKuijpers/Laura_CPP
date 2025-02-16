#ifndef NUMBER_PREDICTOR_H
#define NUMBER_PREDICTOR_H

#include <memory>
#include <vector>
#include <map>
#include "NeuralLayer.h"

struct NumberProbability {
    int number;
    double probability;

    bool operator<(const NumberProbability& other) const {
        return probability > other.probability;  // Sort in descending order
    }
};

struct NumberCombination {
    std::vector<int> numbers;
    double probability;

    bool operator<(const NumberCombination& other) const {
        return probability > other.probability;  // Sort by highest probability
    }
};

class NumberPredictor {
public:
    NumberPredictor();
    void addSequence(const std::vector<int>& numbers);
    std::vector<double> predict();
    std::vector<NumberProbability> predictTop7();
    std::vector<NumberCombination> predictCombinations(int numCombinations = 5);
    void displayCurrentData();
    void save(const std::string& filename) const;
    static std::unique_ptr<NumberPredictor> load(const std::string& filename);
    void trainOnSequence(const std::vector<int>& sequence);
    void testAccuracy(const std::vector<std::vector<int>>& test_sequences);
    void validateAgainstHistory();
    void validateLastPrediction(const std::vector<int>& actual_sequence);
    void showPredictionStats();
private:
    void analyzePairs();
    std::unique_ptr<NeuralLayer> layer1;
    std::unique_ptr<NeuralLayer> layer2;
    std::unique_ptr<NeuralLayer> layer3;
    std::vector<int> frequencies;
    int total_numbers;
    std::map<int, std::vector<int>> sequences;
    std::vector<double> recent_patterns;
    std::vector<double> position_freq;

    double getPatternWeight(int sequence_age) const {
        return std::exp(-0.5 * sequence_age);
    }

    void normalizeVector(std::vector<double>& vec) {
        double sum = 0.0;
        for(double val : vec) {
            sum += val;
        }
        if(sum > 0) {
            for(double& val : vec) {
                val /= sum;
            }
        }
    }

    std::vector<double> getBaseProbabilities(const Matrix& input) {
        // Forward pass through neural network
        Matrix hidden1 = layer1->forward(input);
        Matrix hidden2 = layer2->forward(hidden1);
        Matrix output = layer3->forward(hidden2);

        // Convert to vector
        std::vector<double> probs(52);
        for(int i = 0; i < 52; i++) {
            probs[i] = output.get(0, i);
        }
        normalizeVector(probs);
        return probs;
    }
};

#endif //NUMBER_PREDICTOR_H