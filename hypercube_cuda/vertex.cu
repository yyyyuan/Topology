%%writefile vertex.cu

#include "vertex.cuh"

#include "hypercube_kernel_params.cuh"

__device__ void spin(
  int index,
  int *vertex_neighbor_index) {
    // The first and last neighbor_indexes in the list are connected with each other.
    vertex_neighbor_index[index] = (vertex_neighbor_index[index] + 1) % (22 - 1);
    return;
  }

__device__ void run_vertex(
  int index,
  const HypercubeKernelParams& params) {
    // Special case handling input nodes.
    if (index < params.image_buffer_size) {
      params.excited[index] = true;
      //params.vertex_internal_state[index] = true;

      if (params.clock_tick[index] > 0) {
        if (params.comparessed_image_buffer[index] > 0) {
          params.vertex_internal_state[index] = true;
          params.comparessed_image_buffer[index]--;
        }
        else {
          params.vertex_internal_state[index] = false;
        }
        params.clock_tick[index]--;
      }

      if (params.clock_tick[index] == 0) {
        // For 0-255 values, for now we split it into 8 buckets.
        // This means one image is taking 8 clock cycles to complete.
        // Eventually we can try to increase the count of buckets to even 256
        // to achieve a super high graduality control.
        params.comparessed_image_buffer[index] = params.image_buffer[index] >> 5;
        params.clock_tick[index] = 8; // Reset the clock for image input.
      }
      return;
    }

    // Reset the refractory period if the vertex is in rest.
    // Skip current round.
    if (params.excited[index]) {
        params.excited[index] = false;
        return;
    }

    // TODO: For vertex acceptin inputs from outside, there can be a special handling to make neighbr_address not change or only change in a very small range.
    // In this way, the vertex can keep "spinning" but the spin has no real impact.
    int32_t neighbor_index = params.vertex_neighbor_index[index]; // hypercube_dim
    int32_t neighbor_address = index ^ (1 << neighbor_index);  // Flip the bit at neighbor_index.

    // Only pull data from the neighbor_vertex if it's excited.
    if (params.excited[neighbor_address]) {
        if (params.vertex_internal_state[neighbor_address] == params.vertex_internal_state[index]) {
            params.energy[index]++;

            // The vertex becomes excited when it reaches the upper_excite_thresold.
            // At the initial stage, a simple vertex has 1 energy, 2 upper_threshold and 1 as lower_threshoold.
            //`>=` is good enough to capture this edge case.
            if (params.energy[index] >= params.vertex_upper_bound[index]) {
                params.excited[index] = true;
                // Increase the excite_threshold if the vertex fires.
                int32_t current_upper_excite_threshold = params.vertex_upper_bound[index];
                params.vertex_upper_bound[index] = params.vertex_upper_bound[index] + params.vertex_lower_bound[index];
                params.vertex_lower_bound[index] = current_upper_excite_threshold;
            }
        }
        else {
            params.energy[index]--;

            // Downgrade the excite_threshold if the energy falls below the lower_excite_threshold.
            // If energy < 1, then the energy reaches 0, it will flip and reset the vertex, hence the edge case is covered.
            if (params.energy[index] < params.vertex_lower_bound[index]) {
                // The vertex is also excited when the threshold downgrades due to degrading energy.
                params.excited[index] = true;
                int32_t current_lower_excite_threshold = params.vertex_lower_bound[index];
                params.vertex_lower_bound[index] = params.vertex_upper_bound[index] - params.vertex_lower_bound[index];
                params.vertex_upper_bound[index] = current_lower_excite_threshold;
            }
        }
    }

    // Flip the internal_state if no energy is left.
    // The vertex totally collapses!
    if (params.energy[index] == 0) {
        // The vertex also becomes excited when its status flips.
        params.excited[index] = true;
        params.vertex_internal_state[index] = !params.vertex_internal_state[index];
        params.energy[index] = 1;  // Reset the vertex energy.
        params.vertex_upper_bound[index] = 2;
        params.vertex_lower_bound[index] = 1;
    }

    // if (vertex.type != VertexType::INPUT && vertex.excited) {
    //     printf("Excited vertex!: vertex.internal_state: %d, energy: %d", vertex.internal_state, vertex.energy);
    // }

    // The vertex spins...
    spin(index, params.vertex_neighbor_index);
  }

__device__ void run_vertex_math(
    int index,
    const HypercubeKernelParams& params)
{
    // --- 1. Special Case: Input Nodes ---
    if (index < params.image_buffer_size) {
        params.excited[index] = true;

        if (params.clock_tick[index] > 0) {
            if (params.comparessed_image_buffer[index] > 0) {
                params.vertex_internal_state[index] = true;
                params.comparessed_image_buffer[index]--;
            } else {
                params.vertex_internal_state[index] = false;
            }
            params.clock_tick[index]--;
        }

        if (params.clock_tick[index] == 0) {
            params.comparessed_image_buffer[index] = params.image_buffer[index] >> 5;
            params.clock_tick[index] = 8;
        }
        return;
    }

    // --- 2. Refractory Reset for Normal Nodes ---
    bool is_currently_excited = params.excited[index];
    if (is_currently_excited) {
        params.excited[index] = false;
        return; // Skip cycle during refractory/excited reset
    }

    // --- 3. Compute Hypercube Neighbor Address ---

    // 1. Fetch current node state
    bool S_i = params.vertex_internal_state[index];
    bool E_i = params.excited[index];
    int  U_i = params.energy[index];

    // 2. Compute hypercube neighbor address via XOR bit shift
    int d_i = params.vertex_neighbor_index[index];
    int j   = index ^ (1 << d_i);

    // Fetch neighbor states
    bool S_j = params.vertex_internal_state[j];
    bool E_j = params.excited[j];

    // 3. Mathematical Energy Delta: E_j * (1 - 2 * (S_i ^ S_j))
    // If neighbor not excited, match_delta = 0
    int match_delta = (S_i == S_j) ? 1 : -1;
    int delta_U     = E_j ? match_delta : 0;

    int U_next = U_i + delta_U;

    // 4. Evaluate Threshold Predicates (0 or 1)
    bool fired    = (U_next >= params.vertex_upper_bound[index]);
    bool degraded = (U_next < params.vertex_lower_bound[index]);
    bool collapsed= (U_next == 0);

    // 5. Update Excitation & State without branching
    // If currently excited, force reset to false (!E_i). Else evaluate firing/collapse conditions.
    bool E_next = (!E_i) && (fired || degraded || collapsed);

    // Flip internal state if collapsed (S_i ^ 1)
    bool S_next = collapsed ? !S_i : S_i;

    // Reset energy to 1 if collapsed, else apply U_next
    int U_final = collapsed ? 1 : U_next;

    // 6. Write back to memory
    params.excited[index]               = E_next;
    params.vertex_internal_state[index] = S_next;
    params.energy[index]                = U_final;

    // Advance hypercube dimension modulo 22
    params.vertex_neighbor_index[index] = (d_i + 1) % 22;
}