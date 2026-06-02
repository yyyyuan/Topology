#ifndef _PARTICLE_H_
#define _PARTICLE_H_

// The Basic QM Particle used in topology

#include <cstdint>  // Required for int32_t

#include "constants.h"

struct Particle {
    bool internal_state = true; // True means 1, False means -1.
    int32_t energy = 1; // The energy the particle holds.
    int32_t next_energy_level = 2;  // The energy level that the particle needs to reach to jump to the next state.
    int32_t address = 0;  // The address of the particle in the hypercube.
    bool excited = false; // Telling if the particle can be read by other neighbors.
    bool is_in_refractory_period = false;  // After being excited, entering refractory_period in the next round, where the particle stops any activity.

    // Following 2 values are combined to decide the address of the neighbor.
    bool direction = DIR_INCREASE_K;
    int32_t neighbor_index = 0; 
};

#endif