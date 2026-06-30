#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <cstdint> // Required for int32_t
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct Pattern {
    // Indexes of vertexes that are active with a specific image.
    std::unordered_set<int32_t> vertexes;

    // Indexes of vertexes that are active with a specific image.
    // Weight is calculated by the number of vertex showing up / the count of times where this image category is used.
    std::unordered_map<int32_t, int32_t> pattern_weights;
    // How many rounds did this image category show up in hypercube.
    // This is used to calculate float weights, which is used in deciding the iamge category.
    int32_t count_of_rounds;

    // Vertex energy threshold used in counting pattern vertexes.
    int32_t filter_threshold = 5;

    // The category this pattern indicates.
    // TODO: use std::string instead?
    int32_t category = 0;
};

// Classifier stores image category patterns.
extern std::unordered_map<int32_t, Pattern> classifier;

extern int32_t best_matched_category;
extern float highest_probability_score;

extern std::vector<float> probability_score_list;

// TODO: Calculate if the pattern shown up in the hypercube matches with recorded ones;
//       if not, how off they are.

// Scan through the hypercube and find the pattern matched in the classifier.
// Find the Lowest Common Ancestor (LCA) between the pattern in hypercube and the one in classifier.
//
// If pattern (hypercube) has 1000 active vertexes, and pattern (classifier) has 500 same index vertexes,
// then update the classifier pattern so it contains the shared vertexes in both hypercube and classifier ones.
//
// Before doing this update, calculate how off classifier one is with hypercube one:
// 1. How many are same-index vertexes
// 2. How many are different-index vertexes
// 3. How many vertexes in classifier pattern, and percentage of same-index/diff-index?
// 4. Optional: Is there another category shares the exact same classifier pattern (This means the pattern is not accurate since it covers 2 categories)
void signal_classification(int32_t expected_img_category);

// Calculate the probability score of the expected_img_category in current hypercube.
// Higher score means the pattern inside hypercube matches with the classifier pattern.
float calculate_pattern_probabilty(int32_t expected_img_category);

// Returns the index of the category that has the best match with hypercubee pattern.
// This means it has the highest probability score.
//
// This function is called in training phase.
int32_t find_matched_pattern();

int32_t find_matched_pattern_in_validation();

#endif