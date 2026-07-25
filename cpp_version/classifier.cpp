#include "classifier.h"

#include <cstdint> // Required for int32_t
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
        // if (vertex.energy >= CLASSIFIER_ENERGY_THRESHOLD) {
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

float calculate_pattern_probabilty(int32_t expected_img_category, int32_t& hit_vertex_count, int32_t& excited_vertex_count) {
    if (classifier.find(expected_img_category) == classifier.end()) {
        return 0;
    }

    const Pattern& pattern = classifier.at(expected_img_category);
    float probability_score = 0;
    hypercube_state_snapshot.clear(); // Reset the hypercube state snapshot.
    for (const Vertex& vertex : hypercube_array) {
        if (vertex.type == VertexType::INPUT) {
            continue;
        }

        // Only recognize vertexes which exceed pre-decided energy threshold.
        // TODO:  && vertex.energy >= CLASSIFIER_ENERGY_THRESHOLD
        if (vertex.excited) {
            // Record hypercube state.
            excited_vertex_count++;
            hypercube_state_snapshot.insert(vertex.address);
            if (accumulated_hypercube_state.find(vertex.address) != accumulated_hypercube_state.end()) {
                accumulated_hypercube_state[vertex.address]++;
            }
            else {
                accumulated_hypercube_state[vertex.address] = 1;
            }

            if (pattern.pattern_weights.find(vertex.address) != pattern.pattern_weights.end()) {
                hit_vertex_count++;
                probability_score += static_cast<float>(pattern.pattern_weights.at(vertex.address)) / pattern.count_of_rounds;
            }
        }
        
    }

    return probability_score;
}

int32_t find_matched_pattern(bool verbose) {
    static int col_width = 16;

    if (verbose) {
        std::printf("Classifier Summary\n");
        std::printf("\n===========\n");
        std::cout << "| " << std::setw(col_width) << std::left << "Category Index"
                << " | " << std::setw(col_width) << std::left << "Prob Score"
                << " | " << std::setw(col_width) << std::left << "Match Percentage"
                << " |\n";
    }
    for (int32_t category = 0; category < CATEGORY_COUNT; category++) {
        // if (category == 10) { continue; }
        int32_t hit_vertex_count = 0;
        int32_t excited_vertex_count = 0;
        float calculated_prob_score = calculate_pattern_probabilty(category, hit_vertex_count, excited_vertex_count);
        if (calculated_prob_score > highest_probability_score) {
            best_matched_category = category;
            highest_probability_score = calculated_prob_score;
        }
        probability_score_list[category] += calculated_prob_score;
        accumulated_hypercube_state_array[category][0] += hit_vertex_count;
        accumulated_hypercube_state_array[category][1] += excited_vertex_count;

        if (verbose) {
            std::cout << "| " << std::setw(col_width) << std::right << std::dec << category
                  << " | " << std::setw(col_width) << std::right << calculated_prob_score
                  << " | " << std::setw(col_width) << std::right << (excited_vertex_count == 0 ? 0.0f : static_cast<float>(hit_vertex_count) / excited_vertex_count)
                  << " |\n";
        }
    }

    int max_probability_score_index = 0;
    for (int i = 0; i < probability_score_list.size(); i++) {
        if (probability_score_list[i] > probability_score_list[max_probability_score_index]) {
            max_probability_score_index = i;
        }
    }

    if (verbose) {
        std::printf("Accumulated Classifier Summary\n");
        std::printf("\n===========\n");
        std::cout << "| " << std::setw(col_width) << std::left << "Category Index"
                << " | " << std::setw(col_width) << std::left << "Accumulated Prob Score"
                << " | " << std::setw(col_width) << std::left << "Accumulated Match Percentage"
                << " | " << std::setw(col_width) << std::left << "Hit Vertex Count"
                << " | " << std::setw(col_width) << std::left << "Excited Vertex Count"
                << " |\n";

        for (int i = 0; i < probability_score_list.size(); i++) {
            std::cout << "| " << std::setw(col_width) << std::right << std::dec << i
                    << " | " << std::setw(col_width) << std::right << probability_score_list[i]
                    << " | " << std::setw(col_width) << std::right << (accumulated_hypercube_state_array[i][1] == 0 ? 0.0f : static_cast<float>(accumulated_hypercube_state_array[i][0]) / accumulated_hypercube_state_array[i][1])
                    << " | " << std::setw(col_width) << std::right << accumulated_hypercube_state_array[i][0]
                    << " | " << std::setw(col_width) << std::right << accumulated_hypercube_state_array[i][1]
                    << " |\n";
        }

        std::printf("The best matched category is %d, with prob_score: %f \n", best_matched_category, highest_probability_score);
        std::printf("The best matched category in accumulated probability_score_list is %d, with prob_score: %f \n", max_probability_score_index, probability_score_list[max_probability_score_index]);
    }

    return best_matched_category;
}

void reset_classifier(int32_t img_category) {
    if (classifier.find(img_category) == classifier.end()) {
        return;
    }
    Pattern& pattern = classifier.at(img_category);
    pattern.count_of_rounds = 0;
    
    return;
}
