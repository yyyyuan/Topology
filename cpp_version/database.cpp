#include <cstdint>  // Required for int32_t
#include <vector>

#include "constants.h"
#include "database.h"

// ========== Global array ======
std::vector<int32_t> global_array(1 << ADDR_BITS);
std::vector<int32_t> cycle_delays_array(1 << ADDR_BITS);
std::vector<int32_t> index_array(1 << ADDR_BITS);