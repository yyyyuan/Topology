#include <cstdint>  // Required for int32_t
#include <string>
#include <vector>

#include "constants.h"
#include "database.h"
#include "manifold_operators.h"

// ============ Image loading ===============
// This function currently only loads one image into manifold.
// Hence signals in input nodes don't need to change.
// TODO: Upgrade this function so it can read tons of data from sources such as ImageNet.
// TODO: Change the input formats from constant 1s into 0<->1 pulse to mimic input signals.
void load_image_to_manifold(int32_t input_range);

bool is_input_range(int32_t index);
// ========== End of Image Loading =============

// ========== Manifold Loading & Saving =========
bool save_manifold(const std::vector<int32_t>& manifold_array, const std::string& filepath, const std::string& debug_filepath);

bool load_manifold(std::vector<int32_t>& manifold_array, const std::string& filepath);
// ========== End of Manifold Loading & Saving ============