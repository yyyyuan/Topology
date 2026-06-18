// A new program running the manifold using the new vertex design.
// Putting everything in a new file, meaning a new start I guess.

#include <cstdint> // Required for int32_t
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "constants.h"
#include "database.h"
#include "vertex.h"

void build_input_array() {
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            bool t = (x + y) > IMAGE_HEIGHT;
            input_array[y * IMAGE_WIDTH + x] = t;
        }
    }
}

// Calculate the count of nodes with state 1 in the hypercube.
int32_t analyze_hypercube() {
    std::map<int32_t, int32_t> fibonacci_bucket;
    int32_t hypercube_structure[ADDR_BITS+1][3] = {};  // Each layer contains minimal_energy, maximum_energy, number of nodes with energy >= 2.
    int32_t active_state_count = 0;
    for (Vertex vertex : hypercube_array) {
        if (vertex.internal_state) {
            active_state_count++;
        }
        fibonacci_bucket[vertex.upper_excite_thresold]++;

        // bit counting
        int active_bits = __builtin_popcount(vertex.address);
        if (hypercube_structure[active_bits][0] == 0) {
            hypercube_structure[active_bits][0] = vertex.energy;
        }

        if (vertex.energy > 1 && vertex.type != VertexType::INPUT) {
            hypercube_structure[active_bits][0] = std::min(hypercube_structure[active_bits][0], vertex.energy);
            hypercube_structure[active_bits][1] = std::max(hypercube_structure[active_bits][1], vertex.energy);
            hypercube_structure[active_bits][2]++;
        }
    }

    static int col_width = 16;
    std::printf("\nHypercube structure in the format of Hammer String");
    std::printf("\n===========\n");
    std::cout << "| " << std::setw(col_width) << std::left << "Hammer String"
              << " | " << std::setw(col_width) << std::left << "Minimal Energy"
              << " | " << std::setw(col_width) << std::left << "Maximum Energy"
              << " | " << std::setw(col_width) << std::left << "Count (>= 2)" 
              << " |\n";
    for (int i = 0; i <= ADDR_BITS; ++i) {
        std::cout << "| " << std::setw(col_width) << std::right << i
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i][0]
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i][1]
                  << " | " << std::setw(col_width) << std::right << hypercube_structure[i][2] 
                  << " |\n";
    }
    std::printf("===========\n");

    std::printf("\nEnergy allocation");
    std::printf("\n===========\n");
    for (const auto& [bucket, count] : fibonacci_bucket) {
        std::printf("%d: %d\n", bucket, count);
    }
    std::printf("===========\n");

    // TODO: To better understand hypercube properties, we also want to log the addresses of active vertexes.

    return active_state_count;
}

void record() {
    printf("*** Active vertex count: {%d} ***\n", analyze_hypercube());
}

int main(int argc, char *argv[])
{

    bool use_save_data = false;
    int32_t maximum_runs = 10;
    bool need_init = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--use_save_data")
        {
            use_save_data = true;
        }
        if (arg == "--runs" && i + 1 < argc)
        {
            maximum_runs = static_cast<int32_t>(std::atoi(argv[++i]));
        }
        if (arg == "--init")
        {
            need_init = true;
        }
    }

    build_input_array();
    output_array[0] = false;

    // Initialize the global_array
    for (int i = 0; i < (1 << ADDR_BITS); i++)
    {
        if (need_init)
        {
            // Allocate INPUT/NORMAL/OUTPUT vertexes.
            VertexType type = VertexType::NORMAL;
            if (i < IMAGE_WIDTH * IMAGE_HEIGHT) {
                type = VertexType::INPUT;
            }
            // Select 1000 slots in the middle of hypercube as output slots.
            // TODO: It looks like output slots must be very close to input slots to build internal connections because input range is small.
            if (i >= ((1 << (ADDR_BITS - 10)) - OUTPUT_SIZE) && (i < (1 << (ADDR_BITS - 10)))) {
                type = VertexType::OUTPUT;
            }

            hypercube_array[i] = Vertex{
                .address = i,
                .type = type};
        }
    }
    record();
    printf("output_array status before the run: {%d} \n", output_array[0]);

    int32_t count = 0;
    while (count++ < maximum_runs)
    {
        for (int i = 0; i < hypercube_array.size(); i++)
        {
            // debug(hypercube_array[i]);
            execute(hypercube_array[i]);
        }

        std::cout << "output_array status: {" << output_array[0] << "} \n";

        record();
    }
}