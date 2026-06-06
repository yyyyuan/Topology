#ifndef _VERTEX_H_
#define _VERTEX_H_

// The basic vertex used in the hypercube topology.
// Mimic the QM Particle I designed to unify the Theory of General Relativity and the Quantum Mechanics.
//
// It's called "Vertex" instead of of "Particle" to tell the naunce between this new terminology and the macro scale term "Particle".
// With this new term it tells explicitly that this is something different from the traditional or even QM particles in physics.
//
// The first step to build the new system.

#include <cstdint>  // Required for int32_t
#include <vector>

#include "constants.h"

// The basic unit consisting the hypercube topology.
// With its immaculate properties a new self-evolving system shows up.
struct Vertex {
    bool internal_state = true; // True means 1, False means -1.
    int32_t energy = 1; // The energy held in this vertex.
    int32_t upper_excite_thresold = 2;  // The energy level that the vertex needs to reach to trigger the excite state, 
                                        // which could generate something similar to the "particle" in physics.
    int32_t lower_excite_thresold = 1;  // If the vertex energy is below the lowoer excite threshold, both upper and lower excite threshold need to degrade.
                                        // These two numbers are adjacent Fibonacci numbers.
    int32_t address = 0;  // The address of the vertex in the hypercube.
    bool excited = false; // Telling if the vertex can be read by other neighbors.
    bool is_in_refractory_period = false;  // After being excited, entering refractory_period in the next round, where the vertex stops any activity.

    // Following 2 values are combined to decide the address of the neighbor.
    bool direction = DIR_INCREASE_K;
    int32_t neighbor_index = 0; 
};

int32_t calculate_neighbor_address(int32_t neighbor_index, int32_t address);


// The vertex rotates to the next neighbor.
void spin(Vertex& vertex);

// Let there be Light.
void execute(Vertex& vertex);

// Print out details of the vertex.
void debug(const Vertex& vertex);

#endif