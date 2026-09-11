%%writefile hypercube_classifier.cuh

#ifndef HYPERCUBE_CLASSIFIER_CUH
#define HYPERCUBE_CLASSIFIER_CUH

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// CONFIGURATION PARAMETERS & HYPERCUBE DIMENSIONS
// ============================================================================
constexpr int RAW_FRAMES        = 200;                          // Total raw timeframe snapshots
constexpr int NUM_PATCHES       = 256;                          // Spatial patches (N)
constexpr int WORDS_PER_FRAME   = 512;                          // 32-bit words per frame patch
constexpr int BOOLS_PER_FRAME_PATCH = WORDS_PER_FRAME * 32;     // 16,384 booleans per frame patch

// Raw Hypercube size per frame = 256 patches * 16,384 bools = 4,194,304 booleans (4.19 MB)
constexpr int HYPERCUBE_BOOLS   = NUM_PATCHES * BOOLS_PER_FRAME_PATCH;

constexpr int TUBELET_FRAMES    = 4;                            // 4 raw frames bundled per tubelet step
constexpr int TEMPORAL_STEPS    = RAW_FRAMES / TUBELET_FRAMES;  // T = 50 temporal tubelet steps
constexpr int WORDS_PER_PATCH   = WORDS_PER_FRAME * TUBELET_FRAMES; // 2,048 words per Tubelet patch

constexpr int EMBED_DIM         = 128;                          // Transformer embedding dimension (D)
constexpr int NUM_CLASSES       = 1000;                         // 1,000 downstream classification categories

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// ============================================================================
// 1. RAW SIMULATION BUFFER MUTATION KERNEL
// ============================================================================
__global__ void mutate_raw_hypercube_kernel(
    bool* __restrict__ raw_hypercube, // Shape: [RAW_FRAMES, HYPERCUBE_BOOLS]
    uint32_t seed);

// ============================================================================
// 2. SPATIOTEMPORAL PROJECTION KERNEL
// ============================================================================
__global__ void spatiotemporal_projection_kernel(
    const bool* __restrict__ raw_hypercube, // [RAW_FRAMES, HYPERCUBE_BOOLS]
    const float* __restrict__ W_proj,        // [WORDS_PER_PATCH, EMBED_DIM]
    float*       __restrict__ patch_tokens,  // [TEMPORAL_STEPS, NUM_PATCHES, EMBED_DIM]
    float*       __restrict__ densities);     // [TEMPORAL_STEPS, NUM_PATCHES, WORDS_PER_PATCH]

// ============================================================================
// 3. SPATIAL AGGREGATION KERNEL
// ============================================================================
__global__ void spatial_avg_pool_kernel(
    const float* __restrict__ patch_tokens,     // [TEMPORAL_STEPS, NUM_PATCHES, EMBED_DIM]
    float*       __restrict__ temporal_tokens); // [TEMPORAL_STEPS, EMBED_DIM]

// ============================================================================
// 4. STREAMING TEMPORAL SELF-ATTENTION KERNEL
// ============================================================================
__global__ void temporal_self_attention_kernel(
    const float* __restrict__ temporal_in,   // [TEMPORAL_STEPS, EMBED_DIM]
    const float* __restrict__ W_q,           // [EMBED_DIM, EMBED_DIM]
    const float* __restrict__ W_k,           // [EMBED_DIM, EMBED_DIM]
    const float* __restrict__ W_v,           // [EMBED_DIM, EMBED_DIM]
    float*       __restrict__ temporal_out, // [TEMPORAL_STEPS, EMBED_DIM]
    float*       __restrict__ attn_map_out); // [TEMPORAL_STEPS, TEMPORAL_STEPS]

// ============================================================================
// 5. TEMPORAL AGGREGATION KERNEL
// ============================================================================
__global__ void temporal_avg_pool_kernel(
    const float* __restrict__ temporal_tokens, // [TEMPORAL_STEPS, EMBED_DIM]
    float*       __restrict__ pooled_seq);      // [EMBED_DIM]

// ============================================================================
// 6. LINEAR CLASSIFIER KERNEL
// ============================================================================
__global__ void linear_classifier_kernel(
    const float* __restrict__ pooled_seq, // [EMBED_DIM]
    const float* __restrict__ W_class,    // [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ logits,     // [NUM_CLASSES]
    int embed_dim, int num_classes);

// ============================================================================
// 7. SOFTMAX LOSS & BACKWARD GRADIENT PASSES
// ============================================================================
__global__ void softmax_cross_entropy_kernel(
    const float* __restrict__ logits,      // [NUM_CLASSES]
    const float* __restrict__ pooled_seq,  // [EMBED_DIM]
    const int*   __restrict__ label,
    const float* __restrict__ W_class,     // [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ dL_dpooled,  // [EMBED_DIM]
    float*       __restrict__ dL_dW_class, // [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ loss_out,
    int*         __restrict__ correct_out,
    int embed_dim, int num_classes);

// ----------------------------------------------------------------------------
// Attention Backward Kernel: Compute gradients for W_q, W_k, W_v
// ----------------------------------------------------------------------------
__global__ void temporal_attention_backward_kernel(
    const float* __restrict__ dL_dpooled,     // [EMBED_DIM]
    const float* __restrict__ temporal_in,    // [TEMPORAL_STEPS, EMBED_DIM]
    const float* __restrict__ attn_map,       // [TEMPORAL_STEPS, TEMPORAL_STEPS]
    const float* __restrict__ W_q,            // [EMBED_DIM, EMBED_DIM]
    const float* __restrict__ W_k,            // [EMBED_DIM, EMBED_DIM]
    const float* __restrict__ W_v,            // [EMBED_DIM, EMBED_DIM]
    float*       __restrict__ dL_dW_q,        // [EMBED_DIM, EMBED_DIM]
    float*       __restrict__ dL_dW_k,        // [EMBED_DIM, EMBED_DIM]
    float*       __restrict__ dL_dW_v);        // [EMBED_DIM, EMBED_DIM]

__global__ void spatiotemporal_backward_kernel(
    const float* __restrict__ dL_dpooled,  // [EMBED_DIM]
    const float* __restrict__ densities,   // [TEMPORAL_STEPS, NUM_PATCHES, WORDS_PER_PATCH]
    float*       __restrict__ dL_dW_proj);   // [WORDS_PER_PATCH, EMBED_DIM]

// ============================================================================
// 8. ADAMW OPTIMIZER KERNEL
// ============================================================================
__global__ void adamw_update_kernel(
    float* __restrict__ weights, const float* __restrict__ grad,
    float* __restrict__ m, float* __restrict__ v,
    int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step);

#endif // HYPERCUBE_CLASSIFIER_CUH
