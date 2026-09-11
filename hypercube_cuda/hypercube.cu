%%writefile hypercube.cu

#include "debugging.h"
#include "hypercube_classifier.cuh"
#include "hypercube_kernel_params.cuh"
#include "loading.h"
#include "kernel.cuh"

int main(int argc, char** argv) {
  std::string image_path = (argc > 1) ? argv[1] : "sample_data/sample.JPEG";

  uint8_t* d_raw_rgb = nullptr;
  size_t total_elements_in_image;
  int loading_result = load_image(image_path, d_raw_rgb, total_elements_in_image);

  // Create Non-Blocking CUDA Streams and Sync Event
  // `stream_input` controls blocks responsible for input nodes reading.
  cudaStream_t stream_input, stream_compute;

  int N = 1 << 22;  // 4M elements

  // Initalize parameters used inside kernel function.
  HypercubeKernelParams params;
  params.node_count = N;
  params.image_buffer_size = total_elements_in_image;
  params.image_buffer = d_raw_rgb;

  // The record of eneryg arary on host CPU.
  int* h_energy_buffer;

  // Allocate Unified Memory - accessible from CPU or GPU
  //cudaMallocManaged(&d_raw_rgb, N*sizeof(uint8_t));
  cudaMallocManaged(&params.vertex_internal_state, N*sizeof(bool));
  cudaMallocManaged(&params.excited, N*sizeof(bool));
  cudaMallocManaged(&params.energy, N*sizeof(int));
  cudaMallocManaged(&params.vertex_upper_bound, N*sizeof(int));
  cudaMallocManaged(&params.vertex_lower_bound, N*sizeof(int));
  cudaMallocManaged(&params.vertex_neighbor_index, N*sizeof(int));
  cudaMallocManaged(&params.comparessed_image_buffer, N*sizeof(uint8_t));
  cudaMallocManaged(&params.clock_tick, N*sizeof(uint8_t));

  // Allocate host memory mapped directly into GPU address space
  cudaHostAlloc((void**)&h_energy_buffer, N*sizeof(int), cudaHostAllocMapped);
  cudaHostGetDevicePointer((void**)&params.energy, h_energy_buffer, 0);

  // CUDA Streams
  cudaStream_t stream_mutate_ = nullptr;
  cudaStream_t stream_infer_  = nullptr;
  cudaStream_t stream_train_  = nullptr;

  // Create CUDA Streams
  CUDA_CHECK(cudaStreamCreate(&stream_mutate_));
  CUDA_CHECK(cudaStreamCreate(&stream_infer_));
  CUDA_CHECK(cudaStreamCreate(&stream_train_));

  // ==========
  // HypercubeClassifier init
  // ==========
  // Model Parameters & AdamW State
  float *d_W_proj_ = nullptr, *  = nullptr;
  float *d_m_      = nullptr, *d_v_          = nullptr;

  // Activations
  float *d_tokens_infer_   = nullptr;
  float *d_tokens_train_   = nullptr;
  float *d_densities_train_ = nullptr;
  float *d_dL_dtokens_     = nullptr;

  int step_count_ = 0;

  // 2. Allocate Weights & Optimizer State
  int num_weights = WORDS_PER_PATCH * EMBED_DIM;
  size_t weight_bytes = num_weights * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_W_proj_, weight_bytes));
  CUDA_CHECK(cudaMalloc(&d_dL_dW_proj_, weight_bytes));
  CUDA_CHECK(cudaMalloc(&d_m_, weight_bytes));
  CUDA_CHECK(cudaMalloc(&d_v_, weight_bytes));

  CUDA_CHECK(cudaMemset(d_m_, 0, weight_bytes));
  CUDA_CHECK(cudaMemset(d_v_, 0, weight_bytes));

  // Initialize Projection Weights
  std::vector<float> h_W_init(num_weights);
  for (int i = 0; i < num_weights; ++i) {
      h_W_init[i] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f;
  }
  CUDA_CHECK(cudaMemcpy(d_W_proj_, h_W_init.data(), weight_bytes, cudaMemcpyHostToDevice));

  // 3. Allocate Activations
  CUDA_CHECK(cudaMalloc(&d_tokens_infer_, NUM_PATCHES * EMBED_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tokens_train_, NUM_PATCHES * EMBED_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_densities_train_, NUM_PATCHES * WORDS_PER_PATCH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dL_dtokens_, NUM_PATCHES * EMBED_DIM * sizeof(float)));

  // ==========
  // End of HypercubeClassifier init
  // ==========

  run<<<16, 256>>>(params);

  // === HypercubeClassifier Inference =====
  dim3 fwd_grid(NUM_PATCHES);
  dim3 fwd_block(EMBED_DIM);

  vit_embed_forward_kernel<<<fwd_grid, fwd_block, 0, stream_infer_>>>(
      params.excited, d_W_proj_, d_tokens_infer_, nullptr
  );

  // === End of HypercubeClassifier Inference =====

  // 4. IMPORTANT: Wait for GPU to finish before host reads mapped memory!
  cudaDeviceSynchronize();

  save_to_disk("test.bin", params.energy, 100);

  // Copy the image bits from device memory into host memory.
  uint8_t* h_raw_rgb = new uint8_t[params.image_buffer_size];
  cudaMemcpy(h_raw_rgb, params.image_buffer, params.image_buffer_size, cudaMemcpyDeviceToHost);
  save_to_disk("loaded_image.bin", h_raw_rgb, params.image_buffer_size);

  print_dimension_distribution(params.excited, params.energy, params.vertex_internal_state, 22);

  // Free memory
  delete[] h_raw_rgb;
  cudaFree(params.image_buffer);
  cudaFree(params.vertex_internal_state);
  cudaFree(params.excited);
  cudaFree(params.energy);
  cudaFree(params.vertex_upper_bound);
  cudaFree(params.vertex_neighbor_index);
  cudaFree(params.comparessed_image_buffer);
  cudaFree(params.clock_tick);

  return 0;
}
