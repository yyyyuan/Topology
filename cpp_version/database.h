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

#endif
