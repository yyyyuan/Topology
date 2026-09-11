%%writefile hypercube_kernel_params.cuh

#ifndef _HYPERCUBE_KERNEL_PARAMS_H_
#define _HYPERCUBE_KERNEL_PARAMS_H_

// This struct organizes elements required in hypercube calculation
// and is passed into kernel function for simpler function protocol.
struct HypercubeKernelParams {
  int node_count;
  bool *vertex_internal_state = nullptr;
  bool *excited= nullptr;
  int *energy= nullptr;
  int *vertex_upper_bound= nullptr;
  int *vertex_lower_bound= nullptr;
  int *vertex_neighbor_index= nullptr;

  uint8_t* image_buffer= nullptr;
  int image_buffer_size;

  // This buffer stores how many bits left to be fed into hypercube
  // during one single time image input.
  uint8_t *comparessed_image_buffer= nullptr;
  // Clock ticket for image inputs, recording which round it is.
  uint8_t *clock_tick = nullptr;
};

#endif