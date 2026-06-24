#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <cstdint> // Required for int32_t
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct Pattern {
    // Indexes of vertexes that are active with a specific image.
    std::unordered_set<int> vertexes;

    // Vertex energy threshold used in counting pattern vertexes.
    int32_t filter_threshold = 5;

    // The category this pattern indicates.
    // TODO: use std::string instead?
    int32_t category = 0;
};

// Classifier stores image category patterns.
std::unordered_map<int32_t, Pattern> classifier;

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
void find_pattern(int32_t expected_img_category);

#endif _CLASSIFIER_H_