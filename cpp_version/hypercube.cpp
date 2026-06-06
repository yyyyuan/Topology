// A new program running the manifold using the new vertex design.
// Putting everything in a new file, meaning a new start I guess.

#include <cstdint>  // Required for int32_t
#include <string>
#include <fstream>

#include "constants.h"
#include "database.h"
#include "vertex.h"

int main(int argc, char* argv[]) {
    
    bool use_save_data = false;
    int32_t maximum_runs = 10;
    bool need_init = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--use_save_data") {
            use_save_data = true;
        }
        if (arg == "--runs" && i + 1 < argc) {
            maximum_runs = static_cast<int32_t>(std::atoi(argv[++i]));
        }
        if (arg == "--init") {
            need_init = true;
        }
    }

    // Initialize the global_array
    for (int i = 0; i < (1 << ADDR_BITS); i++) {
        if (need_init) {
            hypercube_array[i] = Vertex{};
        }
    }

    for (int i = 0; i < hypercube_array.size(); i++) {
        debug(hypercube_array[i]);
    }
}