%%writefile kernel.cu

#include <cstdio>
#include <fstream>

#include "hypercube_kernel_params.cuh"
#include "input_node.cuh"
#include "vertex.cuh"

// TODO:
// 4 different sections in this hypercube, they should all be put inside the same kernel function.
// 1. input nodes (image loading should happen at host CPU asynchronously to not block hypercude)
// 2. hypercube nodes (runs asynchronously in a separate CUDA stream)
// 3. output nodes (in real implementation, these can be some sub-cube with dimension k,
//    for example, a 3-D cube can be used to trigger wheel motions of a robot.)
// 4. classification layer (using transformer model and runs asynchronously in a separate CUDA stream)
//
// Summary:
// 2 CUDA streams are required, 1st is for hypercube execution; 2nd is for classification, recognizing if images are really stored.
// Host CPU is used to load images.
// Create some sub-cube for output controls, such as output nodes controlling robot motions.

__global__
void run(HypercubeKernelParams params) {

  //if (blockIdx.x == 0) {
  //  run_input_node();
  //}

  //if (blockIdx.x == 1) {
  //  energy[0]++;
  //}

  //if (blockIdx.x < 2) {
  //  return;
  //}

  // Shift block indexing so compute blocks start at ID 0
  // First 2 blocks are reserved for I/O outputs.
  // TODO: This is actually not needed anymore because
  // the input block is running in a separate thread.
  int compute_block_idx = blockIdx.x;
  int global_thread_id = compute_block_idx * blockDim.x + threadIdx.x;
  int total_compute_thread_in_grid = (gridDim.x) * blockDim.x;

  // Standard printf inside the GPU kernel
  //printf("Hello from GPU Thread %d (compute_block_idx %d, total_compute_thread_in_grid %d)\n",
  //        global_thread_id, compute_block_idx, total_compute_thread_in_grid);

  int count = 0;
  while (count++ < 2000) {
    for (int i = global_thread_id; i < params.node_count; i += total_compute_thread_in_grid) {
      run_vertex(i, params);
      //run_vertex_math(i, params);
    }
  }
  return;
}
