#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xFFFFFFFF

// Hybrid Asynchronous Hypercube Kernel
__global__ void run_hypercube_warp_async(
    volatile int* node_states, // Volatile ensures writes pass through L2 cache
    volatile int* node_steps,  // Tracks iteration/step per node
    int total_nodes,           // e.g., 2^20 nodes (1,048,576)
    int target_steps           // Maximum steps to run per node
) {
    // Calculate global thread and warp identifiers
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;           // Thread index within warp (0..31)
    int global_warp_id = global_thread_id / WARP_SIZE;
    int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    // Grid-stride warp allocation: Each warp processes a set of contiguous 32-node blocks
    for (int node_base = global_warp_id * WARP_SIZE; node_base < total_nodes; node_base += total_warps * WARP_SIZE) {
        
        int node_id = node_base + lane_id;
        if (node_id >= total_nodes) break;

        int local_step = node_steps[node_id];

        // Autonomous loop per node group assigned to this warp
        while (local_step < target_steps) {
            
            // --- STEP 1: READ CURRENT STATE ---
            int current_state = node_states[node_id];

            // --- STEP 2: INTRA-WARP SYNCHRONOUS EXCHANGE (Shfl) ---
            // For hypercube dimensions 0..4 (2^0 to 2^4 = 32 nodes), neighbors live in the SAME warp!
            // We can exchange data directly via registers in 1 clock cycle without VRAM access.
            int dim0_neighbor_state = __shfl_xor_sync(FULL_WARP_MASK, current_state, 1);  // Dimension 0
            int dim1_neighbor_state = __shfl_xor_sync(FULL_WARP_MASK, current_state, 2);  // Dimension 1
            int dim4_neighbor_state = __shfl_xor_sync(FULL_WARP_MASK, current_state, 16); // Dimension 4

            // --- STEP 3: INTER-WARP ASYNCHRONOUS EXCHANGE (VRAM Read) ---
            // For hypercube dimensions >= 5 (nodes across different warps), read from global VRAM.
            // This neighbor might be at a different step count!
            int dim5_neighbor_id = node_id ^ (1 << 5); 
            int dim5_neighbor_state = node_states[dim5_neighbor_id]; 

            // --- STEP 4: STATE TRANSITION LOGIC ---
            // Combine intra-warp synchronous inputs with inter-warp asynchronous inputs
            int next_state = current_state ^ (dim0_neighbor_state & dim5_neighbor_state);
            
            // Non-arithmetic update rule
            next_state = (next_state << 1) | ((next_state >> 31) & 1);

            // Synchronize warp threads before updating VRAM state (ensures clean lockstep step boundaries inside the warp)
            __syncwarp();

            // --- STEP 5: WRITE BACK & MEMORY FENCE ---
            node_states[node_id] = next_state;
            
            // Flush warp writes to L2 cache so other warps across the GPU can see it asynchronously
            __threadfence(); 

            local_step++;
            node_steps[node_id] = local_step;
        }
    }
}