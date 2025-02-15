//
// Created by hadez on 2025/02/12.
//

#include "../headers/NumberPredictor.h"
#include <iostream>

NumberPredictor::NumberPredictor() :
    hidden_layer(std::make_unique<NeuralLayer>(49, 100, 0.01)),
    output_layer(std::make_unique<NeuralLayer>(100, 49, 0.01)),
    frequencies(49, 0),
    total_numbers(0) {}

void NumberPredictor::addSequence(const std::vector<int>& numbers) {
    // Store sequence
    sequences[sequences.size()] = numbers;

    // Update frequencies
    for(int num : numbers) {
        if(num >= 1 && num <= 49) {
            frequencies[num-1]++;
            total_numbers++;
        }
    }
}

std::vector<double> NumberPredictor::predict() {
    // Convert frequencies to probabilities
    Matrix input(1, 49);
    for(int i = 0; i < 49; i++) {
        input.set(0, i, frequencies[i] / (double)total_numbers);
    }

    // Forward pass
    Matrix hidden = hidden_layer->forward(input);
    Matrix output = output_layer->forward(hidden);

    // Convert output to probabilities
    std::vector<double> probabilities(49);
    for(int i = 0; i < 49; i++) {
        probabilities[i] = output.get(0, i);
    }

    return probabilities;
}

void NumberPredictor::displayCurrentData() {
    std::cout << "\nCurrent Data Analysis:\n";
    std::cout << "----------------------\n";

    // Display frequency counts
    std::cout << "Number frequencies:\n";
    for(int i = 0; i < frequencies.size(); i++) {
        if(frequencies[i] > 0) {
            std::cout << "Number " << (i+1) << ": " << frequencies[i]
                     << " times ("
                     << (frequencies[i] * 100.0 / total_numbers)
                     << "%)\n";
        }
    }

    // Display sequences
    std::cout << "\nStored sequences:\n";
    for(const auto& seq : sequences) {
        std::cout << "Sequence " << seq.first << ": ";
        for(int num : seq.second) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }

    // Display most common patterns
    std::cout << "\nCommon patterns:\n";
    analyzePairs();
}

void NumberPredictor::analyzePairs() {
    std::map<std::pair<int, int>, int> pair_frequencies;

    // Count consecutive number pairs
    for(const auto& seq : sequences) {
        for(size_t i = 0; i < seq.second.size() - 1; i++) {
            pair_frequencies[{seq.second[i], seq.second[i+1]}]++;
        }
    }

    // Display most common pairs
    std::cout << "Most common consecutive pairs:\n";
    for(const auto& pair : pair_frequencies) {
        if(pair.second > 1) {  // Show pairs that appear more than once
            std::cout << pair.first.first << "->" << pair.first.second
                     << ": " << pair.second << " times\n";
        }
    }
}