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
int32_t calcualte_active_vertexes() {
    std::map<int32_t, int32_t> fibonacci_bucket;
    int32_t active_state_count = 0;
    for (Vertex vertex : hypercube_array) {
        if (vertex.internal_state) {
            active_state_count++;
        }
        fibonacci_bucket[vertex.upper_excite_thresold]++;
    }

    for (const auto& [bucket, count] : fibonacci_bucket) {
        std::printf("Energy allocation: %d: %d\n", bucket, count);
    }

    return active_state_count;
}

void record() {
    printf("*** Active vertex count: {%d} ***\n", calcualte_active_vertexes());
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
            VertexType type = VertexType::NORMAL;
            if (i < IMAGE_WIDTH * IMAGE_HEIGHT) {
                type = VertexType::INPUT;
            }
            // Select 1000 slots in the middle of hypercube as output slots.
            // TODO: It looks like output slots must be very close to input slots to build internal connections since input range is small.
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

    // TODO: Define input and output vertexes and put them inside this hypercube.
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