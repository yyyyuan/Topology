#ifndef _DATABASE_H_
#define _DATABASE_H_

#include <cstdint>  // Required for int32_t
#include <vector>

#include "constants.h"
#include "vertex.h"

// ========== Global array ======
extern std::vector<int32_t> global_array;
extern std::vector<int32_t> cycle_delays_array;
extern std::vector<int32_t> index_array;

extern std::vector<Vertex> hypercube_array;
extern std::vector<int32_t> input_array;  // This array represents translated input signals from external world.
extern std::vector<bool> output_array;  // This array represents translated output signals from hypercube.

inline int32_t (*input_array_ptr)[RGB_INPUT_BUFFER_SIZE] = nullptr;  // This array represents translated input signals from external world.
// extern std::vector<std::vector<int32_t>> input_buffer;
inline int32_t input_buffer[CATEGORY_COUNT][TARGET_HEIGHT * TARGET_WIDTH];

// The RGB image buffer has 3 dimensions:
// 1. The number of images
// 2. The number of retina nodes
// 3. 1-bit signal representing part of compressed 8-bit from teh 128 range:
//    8-bits to represent 128 range.
//    R intensity: 0-15 => 00000000
//                 16-31 => 10000000
//                 ...
//                 112-127 => 11111111
inline int32_t rgb_img_buffer[CATEGORY_COUNT][8][RGB_INPUT_BUFFER_SIZE];
// inline int32_t rgb_img_buffer[CATEGORY_COUNT][8][3][RGB_INPUT_BUFFER_SIZE];

// This array represents the images used in valdiation.
// Testing if the trained/interfered hypercube is able to recognize image category.
extern std::vector<std::vector<int32_t>> validation_img_buffer;

#endif
