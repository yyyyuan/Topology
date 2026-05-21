#include "loading.h"

#include <string>
#include <vector>
#include <fstream>

int32_t IDX_DIMENSION_0_RANGE = IMAGE_HEIGHT * IMAGE_WIDTH / 2;
int32_t IDX_DIMENSION_1_RANGE = (1 << (ADDR_BITS - 1)) + IDX_DIMENSION_0_RANGE; // 2^(ADDR_BITS - 1) + Range


// This function currently only loads one image into manifold.
// Hence signals in input nodes don't need to change.
// TODO: Upgrade this function so it can read tons of data from sources such as ImageNet.
// TODO: Change the input formats from constant 1s into 0<->1 pulse to mimic input signals.
void load_image_to_manifold(int32_t input_range) {
  // Hardcode the image in path "/content/test_image.png" into manifold to simplify the POC.
  int32_t idx_dimension0 = 0;
  int32_t idx_dimension1 = global_array.size() / 2; // Idx must be integer
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    for (int x = 0; x < IMAGE_WIDTH; x++) {
      if ((idx_dimension0 + idx_dimension1 - (global_array.size() / 2)) > input_range) {
        break;
      }
      float t = (x + y) / 126.0;
      int32_t r = int(255 * (1 - t));
      int32_t g = int(255 * std::min(t, 1 - t) * 2);
      int32_t b = int(255 * t);

      bool is_dimension_0 = x < IMAGE_WIDTH / 2;

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (r >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (r >> bit_pos) & 1);
          idx_dimension1++;
        }
      }

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (g >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (g >> bit_pos) & 1);
          idx_dimension1++;
        }
      }

      for (int bit_pos = 7; bit_pos >= 0; bit_pos--) {
        if (is_dimension_0) {
          global_array[idx_dimension0] = pack_word(idx_dimension0, 1, 0, 1, (b >> bit_pos) & 1);
          idx_dimension0++;
        } else {
          global_array[idx_dimension1] = pack_word(idx_dimension1, 1, 0, 1, (b >> bit_pos) & 1);
          idx_dimension1++;
        }
      }
    }
  }
  IDX_DIMENSION_0_RANGE = idx_dimension0 + 1;
  IDX_DIMENSION_1_RANGE = idx_dimension1 + 1;
  return;
}

bool is_input_range(int32_t index) {
  if ((index >= 0 && index < IDX_DIMENSION_0_RANGE) ||
      (index >= (global_array.size() / 2) && index < IDX_DIMENSION_1_RANGE)) {
      return true;
  }
  return false;
}

// ========== Manifold Loading & Saving =========
bool save_manifold(const std::vector<int32_t>& manifold_array, const std::string& filepath, const std::string& debug_filepath) {
    std::ofstream out(filepath, std::ios::out | std::ios::trunc);
    std::ofstream debug(debug_filepath, std::ios::out | std::ios::trunc);
    if (!out.is_open() || !debug.is_open()) return false;
    for (int32_t n : manifold_array) {
        out << n << '\n';
        debug << std::bitset<32>(static_cast<uint32_t>(n)) << '\n';
    }
    return out.good();
}

bool load_manifold(std::vector<int32_t>& manifold_array, const std::string& filepath) {
    std::vector<int32_t> input_array;
    std::ifstream in(filepath);
    if (!in.is_open()) return false;
    double value;
    while (in >> value) {
        input_array.push_back(value);
    }

    if (input_array.size() != (1 << ADDR_BITS)) {
      return false;
    }

    manifold_array = std::move(input_array);
    return true;
}

// ========== End of Manifold Loading & Saving ============