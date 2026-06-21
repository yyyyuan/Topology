#include "vertex.h"

#include "constants.h"
#include "database.h"

int32_t calculate_neighbor_address(int32_t neighbor_index, int32_t address)
{
    return address ^ (1 << neighbor_index); // Flip the bit at neighbor_index.
}

void spin(Vertex &vertex)
{
    if (vertex.neighbor_index == 0 && vertex.direction != DIR_INCREASE_K) {
        vertex.direction = DIR_INCREASE_K;
    }
    if (vertex.neighbor_index == (ADDR_BITS - 1) && vertex.direction == DIR_INCREASE_K) {
        vertex.direction = DIR_DECREASE_K;
    }

    if (vertex.direction == DIR_INCREASE_K) {
        vertex.neighbor_index = (vertex.neighbor_index + 1) & MASK_K;
    }
    else {
        vertex.neighbor_index = (vertex.neighbor_index - 1) & MASK_K;
    }

    return;
}

void execute(Vertex &vertex)
{
    // Reset the refractory period if the vertex is in rest.
    // Skip current round.
    if (vertex.excited) {
        vertex.is_in_refractory_period = false;
        vertex.excited = false;
        return;
    }

    Vertex neighbor_vertex;
    if (vertex.type == VertexType::INPUT) {
        // This neighbor_vertex is a virtual input vertex representing signals from external world.
        neighbor_vertex = get_input(vertex);
    }
    else {
        // TODO: For vertex acceptin inputs from outside, there can be a special handling to make neighbr_address not change or only change in a very small range.
        // In this way, the vertex can keep "spinning" but the spin has no real impact.
        int32_t neighbor_address = vertex.address ^ (1 << vertex.neighbor_index); // Flip the bit at neighbor_index.
        neighbor_vertex = hypercube_array.at(neighbor_address);
    }

    // Only pull data from the neighbor_vertex if it's excited.
    if (neighbor_vertex.excited) {
        if (neighbor_vertex.internal_state == vertex.internal_state) {
            vertex.energy++;

            // The vertex becomes excited when it reaches the upper_excite_thresold.
            // At the initial stage, a simple vertex has 1 energy, 2 upper_threshold and 1 as lower_threshoold.
            //`>=` is good enough to capture this edge case.
            if (vertex.energy >= vertex.upper_excite_thresold) {
                vertex.excited = true;
                // Increase the excite_threshold if the vertex fires.
                int32_t current_upper_excite_threshold = vertex.upper_excite_thresold;
                vertex.upper_excite_thresold = vertex.upper_excite_thresold + vertex.lower_excite_thresold;
                vertex.lower_excite_thresold = current_upper_excite_threshold;
            }
        }
        else {
            vertex.energy--;

            // Downgrade the excite_threshold if the energy falls below the lower_excite_threshold.
            // If energy < 1, then the energy reaches 0, it will flip and reset the vertex, hence the edge case is covered.
            if (vertex.energy < vertex.lower_excite_thresold) {
                int32_t current_lower_excite_threshold = vertex.lower_excite_thresold;
                vertex.lower_excite_thresold = vertex.upper_excite_thresold - vertex.lower_excite_thresold;
                vertex.upper_excite_thresold = current_lower_excite_threshold;
            }
        }
    }

    // Flip the internal_state if no energy is left.
    // The vertex totally collapses!
    if (vertex.energy == 0) {
        vertex.internal_state = !vertex.internal_state;
        vertex.energy = 1;  // Reset the vertex energy.
        vertex.upper_excite_thresold = 2;
        vertex.lower_excite_thresold = 1;
    }

    // The vertex spins...
    spin(vertex);

    // OUTPUT vertexes are responsible for writing data to external world.
    if (vertex.type == VertexType::OUTPUT) {
        send_output(vertex);
    }

    // TODO: For vertex that generates outputs and connecting with the outside world, 2 things can be done:
    // 1. Control the range the vertex can spin, so it won't pull data from out transimitter vertex.
    // 2. Add a special way writing outputs into output hardwares (wheel controller, autopilot, microphone etc)
}

void debug(const Vertex& vertex) {
    printf("\ninternal state is: %d", vertex.internal_state);
}

Vertex get_input(const Vertex& input_vertex) {
    Vertex virtual_ipnut_vertex;

    // Only 2 fields are used:
    // 1. excited
    // 2. internal_state

    // For now, the VIV is always marked as excited representing continuous stimulations from external world.
    virtual_ipnut_vertex.excited = true;

    // Copy input signals directly from the input_array following the 1-to-1 address mapping.
    // TODO: Eventually, the behavior should be more natural, meaning that each vertex is able to listen to multiple input slots,
    //       instead of focusing on one slot.
    virtual_ipnut_vertex.internal_state = (*input_array_ptr)[input_vertex.address];

    return virtual_ipnut_vertex;
}

void send_output(const Vertex& output_vertex) {
    // `100` is a temporary threshold.
    if (output_vertex.energy >= 2) {
        // Write a signal into output hardwares.
        // 2026-06-17: For now assume there is only one output slot.
        //
        // When there are more output slots, the output_vertex addresses will be used to decide which vertex writes data to which output slot.
        output_array[0] = true;
    }
}