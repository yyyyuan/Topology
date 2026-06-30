#include "classifier.h"

#include <iomanip>
#include <iostream>
#include <string>

#include "database.h"

void signal_classification(int32_t expected_img_category) {
    if (classifier.find(expected_img_category) == classifier.end()) {
         classifier[expected_img_category] = Pattern{};
    }
    Pattern& pattern = classifier.at(expected_img_category);

    Pattern current_pattern;
    for (const Vertex& vertex : hypercube_array) {
        if (vertex.type == VertexType::INPUT) {
            continue;
        }

        // Only recognize vertexes which exceed pre-decided energy threshold.
        // TODO:  && vertex.energy >= CLASSIFIER_ENERGY_THRESHOLD
        if (vertex.excited) {
            current_pattern.vertexes.insert(vertex.address);
            // pattern.pattern_weights[vertex.address] += vertex.energy;
            pattern.pattern_weights[vertex.address]++;  // TODO: Also test cases where energy is not used in pattern calculatoin.
        }
    }
    pattern.count_of_rounds++;

    // Inner join patterns between current one in hypercube and the recorded one in classifier.
    pattern.vertexes.clear();
    for (const int& num : current_pattern.vertexes) {
        if (pattern.vertexes.find(num) != pattern.vertexes.end()) { // Use .find() != .end() if prior to C++20
            pattern.vertexes.insert(num);
        }
    }
}

float calculate_pattern_probabilty(int32_t expected_img_category) {
    if (classifier.find(expected_img_category) == classifier.end()) {
        return 0;
    }

    const Pattern& pattern = classifier.at(expected_img_category);
    float probability_score = 0;
    for (const Vertex& vertex : hypercube_array) {
        if (vertex.type == VertexType::INPUT) {
            continue;
        }

        // Only recognize vertexes which exceed pre-decided energy threshold.
        // TODO:  && vertex.energy >= CLASSIFIER_ENERGY_THRESHOLD
        if (vertex.excited) {
            if (pattern.pattern_weights.find(vertex.address) != pattern.pattern_weights.end()) {
                probability_score += static_cast<float>(pattern.pattern_weights.at(vertex.address)) / pattern.count_of_rounds;
            }
        }
        
    }

    return probability_score;
}

int32_t find_matched_pattern() {
    static int col_width = 16;
    std::printf("Classifier Summary\n");
    std::printf("\n===========\n");
    std::cout << "| " << std::setw(col_width) << std::left << "Category Index"
              << " | " << std::setw(col_width) << std::left << "Prob Score"
              << " |\n";
    for (int32_t category = 0; category < CATEGORY_COUNT; category++) {
        if (category == 10) { continue; }
        float calculated_prob_score = calculate_pattern_probabilty(category);
        if (calculated_prob_score > highest_probability_score) {
            best_matched_category = category;
            highest_probability_score = calculated_prob_score;
        }
        probability_score_list[category] += calculated_prob_score;

        std::cout << "| " << std::setw(col_width) << std::right << std::dec << category
                  << " | " << std::setw(col_width) << std::right << calculated_prob_score
                  << " |\n";
    }

    int max_probability_score_index = 0;
    for (int i = 1; i < probability_score_list.size(); i++) {
        if (probability_score_list[i] > probability_score_list[max_probability_score_index]) {
            max_probability_score_index = i;
        }
    }

    std::printf("The best matched category is %d, with prob_score: %f \n", best_matched_category, highest_probability_score);
    std::printf("The best matched category in probability_score_list is %d, with prob_score: %f \n", max_probability_score_index, probability_score_list[max_probability_score_index]);

    // std::printf("\nClassifier Details");
    // std::printf("\n===========\n");
    // std::cout << "| " << std::setw(col_width) << std::left << "Img Category"
    //           << " | " << std::setw(col_width) << std::left << "Vertex Address"
    //           << " | " << std::setw(col_width) << std::left << "Count"
    //           << " |\n";
    // for (const auto& classifier_pattern : classifier) {
    //     for (auto& pattern_node : classifier_pattern.second.pattern_weights) {
    //          std::cout << "| " << std::setw(col_width) << std::right << std::dec << classifier_pattern.first
    //               << " | " << std::setw(col_width) << std::right << pattern_node.first
    //               << " | " << std::setw(col_width) << std::right << static_cast<float>(pattern_node.second) / classifier_pattern.second.count_of_rounds
    //               << " |\n";
    //     }
    // }
    // std::printf("===========\n");

    return best_matched_category;
}
