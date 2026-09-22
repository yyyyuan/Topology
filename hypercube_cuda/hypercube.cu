%%writefile hypercube.cu

#include <cstdint>
#include <cstddef>

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

    // =================================================================
    // HypercubeClassifier init
    // =================================================================
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Multi-Timeframe Spatio-Temporal Hypercube Classifier Pipeline        " << std::endl;
    std::cout << "======================================================================" << std::endl;

    bool* d_raw_hypercube = nullptr;
    size_t raw_bytes = static_cast<size_t>(RAW_FRAMES) * HYPERCUBE_BOOLS * sizeof(bool);
    CUDA_CHECK(cudaMalloc(&d_raw_hypercube, raw_bytes));

    int h_target_label = 42;
    int* d_target_label = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target_label, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_target_label, &h_target_label, sizeof(int), cudaMemcpyHostToDevice));

    int num_proj_weights  = WORDS_PER_PATCH * EMBED_DIM;
    int num_class_weights = EMBED_DIM * NUM_CLASSES;
    int num_attn_weights  = EMBED_DIM * EMBED_DIM;

    float *d_W_proj = nullptr, *d_dL_dW_proj = nullptr, *d_m_proj = nullptr, *d_v_proj = nullptr;
    float *d_W_class = nullptr, *d_dL_dW_class = nullptr, *d_m_class = nullptr, *d_v_class = nullptr;

    // Attention Weights, Gradients, and Optimizer States
    float *d_W_q = nullptr, *d_dL_dW_q = nullptr, *d_m_q = nullptr, *d_v_q = nullptr;
    float *d_W_k = nullptr, *d_dL_dW_k = nullptr, *d_m_k = nullptr, *d_v_k = nullptr;
    float *d_W_v = nullptr, *d_dL_dW_v = nullptr, *d_m_v = nullptr, *d_v_v = nullptr;

    CUDA_CHECK(cudaMalloc(&d_W_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_proj, num_proj_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_class, num_class_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_q, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_q, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_q, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_q, num_attn_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_k, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_k, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_k, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_k, num_attn_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_v, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_v, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_v, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_v, num_attn_weights * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_proj, 0, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_proj, 0, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_class, 0, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_class, 0, num_class_weights * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_q, 0, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_q, 0, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_k, 0, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_k, 0, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_v, 0, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_v, 0, num_attn_weights * sizeof(float)));

    std::vector<float> h_W_proj(num_proj_weights), h_W_class(num_class_weights), h_W_attn(num_attn_weights);
    for (int i = 0; i < num_proj_weights; ++i) h_W_proj[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < num_class_weights; ++i) h_W_class[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < num_attn_weights; ++i) h_W_attn[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;

    CUDA_CHECK(cudaMemcpy(d_W_proj, h_W_proj.data(), num_proj_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_class, h_W_class.data(), num_class_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_q, h_W_attn.data(), num_attn_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_k, h_W_attn.data(), num_attn_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_v, h_W_attn.data(), num_attn_weights * sizeof(float), cudaMemcpyHostToDevice));

    float *d_patch_tokens = nullptr, *d_densities = nullptr;
    float *d_temporal_tokens = nullptr, *d_attn_temporal_out = nullptr, *d_attn_map = nullptr;
    float *d_pooled_seq = nullptr, *d_logits = nullptr;
    float *d_dL_dpooled = nullptr, *d_loss_out = nullptr;
    int*   d_correct_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_patch_tokens, TEMPORAL_STEPS * NUM_PATCHES * EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_densities, TEMPORAL_STEPS * NUM_PATCHES * WORDS_PER_PATCH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temporal_tokens, TEMPORAL_STEPS * EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_temporal_out, TEMPORAL_STEPS * EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_map, TEMPORAL_STEPS * TEMPORAL_STEPS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pooled_seq, EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dpooled, EMBED_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_loss_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct_out, sizeof(int)));

    std::cout << "[+] System Initialized:" << std::endl;
    std::cout << "    - raw_hypercube Shape: [" << RAW_FRAMES << ", " << HYPERCUBE_BOOLS << "] ("
                << (raw_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "    - Tubelets (T): " << TEMPORAL_STEPS << " steps x " << NUM_PATCHES << " patches" << std::endl;

    dim3 fwd_proj_grid(NUM_PATCHES, TEMPORAL_STEPS);
    dim3 fwd_proj_block(EMBED_DIM);

    int total_raw_bools = RAW_FRAMES * HYPERCUBE_BOOLS;
    int mutate_blocks = (total_raw_bools + 255) / 256;

    int opt_proj_blocks  = (num_proj_weights + 255) / 256;
    int opt_class_blocks = (num_class_weights + 255) / 256;
    int opt_attn_blocks  = (num_attn_weights + 255) / 256;

    dim3 attn_back_grid((EMBED_DIM + 15) / 16, (EMBED_DIM + 15) / 16);
    dim3 attn_back_block(16, 16);

    // =================================================================
    // End of HypercubeClassifier init
    // =================================================================

    run<<<16, 256>>>(params);

    // === HypercubeClassifier Inference =====
    dim3 fwd_grid(NUM_PATCHES);
    dim3 fwd_block(EMBED_DIM);

    // Forward Pass
    spatiotemporal_projection_kernel<<<fwd_proj_grid, fwd_proj_block>>>(
        d_raw_hypercube, d_W_proj, d_patch_tokens, d_densities
    );

    spatial_avg_pool_kernel<<<TEMPORAL_STEPS, EMBED_DIM>>>(
        d_patch_tokens, d_temporal_tokens
    );

    temporal_self_attention_kernel<<<TEMPORAL_STEPS, EMBED_DIM>>>(
        d_temporal_tokens, d_W_q, d_W_k, d_W_v, d_attn_temporal_out, d_attn_map
    );

    temporal_avg_pool_kernel<<<(EMBED_DIM + 255) / 256, 256>>>(
        d_attn_temporal_out, d_pooled_seq
    );

    linear_classifier_kernel<<<(NUM_CLASSES + 255) / 256, 256>>>(
        d_pooled_seq, d_W_class, d_logits, EMBED_DIM, NUM_CLASSES
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
    // =================================================================
    // Cleanup HypercubeClassifier init
    // =================================================================
    CUDA_CHECK(cudaFree(d_raw_hypercube)); CUDA_CHECK(cudaFree(d_target_label));
    CUDA_CHECK(cudaFree(d_W_proj)); CUDA_CHECK(cudaFree(d_dL_dW_proj));
    CUDA_CHECK(cudaFree(d_m_proj)); CUDA_CHECK(cudaFree(d_v_proj));
    CUDA_CHECK(cudaFree(d_W_class)); CUDA_CHECK(cudaFree(d_dL_dW_class));
    CUDA_CHECK(cudaFree(d_m_class)); CUDA_CHECK(cudaFree(d_v_class));

    CUDA_CHECK(cudaFree(d_W_q)); CUDA_CHECK(cudaFree(d_dL_dW_q)); CUDA_CHECK(cudaFree(d_m_q)); CUDA_CHECK(cudaFree(d_v_q));
    CUDA_CHECK(cudaFree(d_W_k)); CUDA_CHECK(cudaFree(d_dL_dW_k)); CUDA_CHECK(cudaFree(d_m_k)); CUDA_CHECK(cudaFree(d_v_k));
    CUDA_CHECK(cudaFree(d_W_v)); CUDA_CHECK(cudaFree(d_dL_dW_v)); CUDA_CHECK(cudaFree(d_m_v)); CUDA_CHECK(cudaFree(d_v_v));

    CUDA_CHECK(cudaFree(d_patch_tokens)); CUDA_CHECK(cudaFree(d_densities));
    CUDA_CHECK(cudaFree(d_temporal_tokens)); CUDA_CHECK(cudaFree(d_attn_temporal_out));
    CUDA_CHECK(cudaFree(d_attn_map)); CUDA_CHECK(cudaFree(d_pooled_seq));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_dL_dpooled));
    CUDA_CHECK(cudaFree(d_loss_out)); CUDA_CHECK(cudaFree(d_correct_out));

    // =================================================================
    // End of Cleanup HypercubeClassifier init
    // =================================================================



    // =================================================================
    // End of Cleanup Hypercube parameters
    // =================================================================
    delete[] h_raw_rgb;
    cudaFree(params.image_buffer);
    cudaFree(params.vertex_internal_state);
    cudaFree(params.excited);
    cudaFree(params.energy);
    cudaFree(params.vertex_upper_bound);
    cudaFree(params.vertex_neighbor_index);
    cudaFree(params.comparessed_image_buffer);
    cudaFree(params.clock_tick);
    // =================================================================
    // End of Cleanup Hypercube parameters
    // =================================================================

    return 0;
}
