#include <cstdint>  // Required for int32_t
#include <vector>

#include "constants.h"
#include "database.h"

// ========== Global array ======
std::vector<int32_t> global_array(1 << ADDR_BITS);
std::vector<int32_t> cycle_delays_array(1 << ADDR_BITS);
std::vector<int32_t> index_array(1 << ADDR_BITS);

std::vector<Vertex> hypercube_array(1 << ADDR_BITS);
std::vector<int32_t> input_array(IMAGE_HEIGHT * IMAGE_WIDTH);  // For now assume this is only black and white pic.
std::vector<bool> output_array(1);  // The size of output_array can be adjusted.

std::vector<std::vector<int32_t>> input_buffer(2, std::vector<int32_t>(IMAGE_WIDTH * IMAGE_HEIGHT));