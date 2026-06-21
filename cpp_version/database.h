#ifndef _DATABASE_H_
#define _DATABASE_H_

#include <cstdint>  // Required for int32_t
#include <vector>

#include "vertex.h"

// ========== Global array ======
extern std::vector<int32_t> global_array;
extern std::vector<int32_t> cycle_delays_array;
extern std::vector<int32_t> index_array;

extern std::vector<Vertex> hypercube_array;
extern std::vector<int32_t> input_array;  // This array represents translated input signals from external world.
extern std::vector<bool> output_array;  // This array represents translated output signals from hypercube.

inline std::vector<int32_t>* input_array_ptr = nullptr;  // This array represents translated input signals from external world.
extern std::vector<std::vector<int32_t>> input_buffer;

#endif
