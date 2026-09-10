%%writefile main.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

// ============================================================================
// CONFIGURATION PARAMETERS & HYPERCUBE DIMENSIONS
// ============================================================================
// Raw Simulation Buffer: 200 Frames x 256 Patches x 512 Words (104,857,600 Bytes / 104.85 MB)
// Total booleans in our hypercube manifold (4 MB bitfield = 33,554,432 bits)
// Represented in GPU memory as 4,194,304 contiguous bools (1 byte per bool for fast execution)
constexpr int HYPERCUBE_BOOLS = 4194304; 

constexpr int RAW_FRAMES        = 200;
constexpr int TUBELET_FRAMES    = 4;                            // 4 raw frames bundled per tubelet step
constexpr int TEMPORAL_STEPS    = RAW_FRAMES / TUBELET_FRAMES;  // T = 50 temporal tubelet steps
constexpr int NUM_PATCHES       = 256;                          // Spatial patches (N)
constexpr int WORDS_PER_FRAME   = 512;                          // 32-bit words per frame patch
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
// Fills/mutates raw 200-frame boolean hypercube state [RAW_FRAMES, NUM_PATCHES, WORDS_PER_FRAME * 32]
__global__ void mutate_raw_hypercube_kernel(
    bool* __restrict__ sequence_hypercube,
    uint32_t seed) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bools = RAW_FRAMES * NUM_PATCHES * WORDS_PER_FRAME * 32;
    if (tid >= total_bools) return;

    uint32_t x = tid ^ seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    sequence_hypercube[tid] = (x & 0x1) ? true : false;
}

// ============================================================================
// 2. SPATIOTEMPORAL PROJECTION KERNEL (Tubelet Slicing + Projection)
// ============================================================================
// Slices raw buffer into T=50 Tubelets (4 frames x 512 words = 2,048 words)
// Map: [50 Steps, 256 Patches, 2048 Words] -> [50 Steps, 256 Tokens, 128 Features]
__global__ void spatiotemporal_projection_kernel(
    const bool* __restrict__ raw_hypercube, // [200, 256, 512 * 32] bools
    const float* __restrict__ W_proj,        // [2048, 128]
    float*       __restrict__ patch_tokens,  // [50, 256, 128]
    float*       __restrict__ densities)     // [50, 256, 2048]
{
    int step_idx  = blockIdx.y; // 0 ... 49 (Temporal Tubelet Step)
    int patch_idx = blockIdx.x; // 0 ... 255 (Spatial Patch)
    int dim_idx   = threadIdx.x; // 0 ... 127 (Embedding Dim)

    if (step_idx >= TEMPORAL_STEPS || patch_idx >= NUM_PATCHES || dim_idx >= EMBED_DIM) return;

    int token_offset = (step_idx * NUM_PATCHES + patch_idx);
    float total_projection = 0.0f;

    // Aggregate across 4 raw frames within this tubelet step
    for (int tube_f = 0; tube_f < TUBELET_FRAMES; ++tube_f) {
        int raw_frame_idx = step_idx * TUBELET_FRAMES + tube_f;
        const bool* frame_patch_bools = raw_hypercube + 
            (raw_frame_idx * NUM_PATCHES + patch_idx) * (WORDS_PER_FRAME * 32);

        const uchar4* vec_ptr = reinterpret_cast<const uchar4*>(frame_patch_bools);
        int num_vecs = (WORDS_PER_FRAME * 32) / 4; // 4096 uchar4s per patch frame

        for (int i = 0; i < num_vecs; ++i) {
            int word_in_frame = i / 8;
            int word_in_tubelet = tube_f * WORDS_PER_FRAME + word_in_frame;

            if (densities != nullptr && dim_idx == 0 && (i % 8 == 0)) {
                float density_sum = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    uchar4 sub_b4 = vec_ptr[i + k];
                    density_sum += (sub_b4.x ? 1.0f : 0.0f) + (sub_b4.y ? 1.0f : 0.0f) +
                                   (sub_b4.z ? 1.0f : 0.0f) + (sub_b4.w ? 1.0f : 0.0f);
                }
                // Store density in tubelet index [0 ... 2047]
                densities[token_offset * WORDS_PER_PATCH + word_in_tubelet] = density_sum / 32.0f;
            }

            uchar4 b4 = vec_ptr[i];
            float v = (b4.x ? 1.0f : 0.0f) + (b4.y ? 1.0f : 0.0f) +
                      (b4.z ? 1.0f : 0.0f) + (b4.w ? 1.0f : 0.0f);

            total_projection += v * 0.25f * W_proj[word_in_tubelet * EMBED_DIM + dim_idx];
        }
    }

    patch_tokens[token_offset * EMBED_DIM + dim_idx] = total_projection;
}

// ============================================================================
// 3. SPATIAL AGGREGATION KERNEL
// ============================================================================
// Collapses spatial patches via global average pooling across N=256 patches.
// Map: [50 Steps, 256 Tokens, 128 Features] -> [50 Temporal Tokens, 128 Features]
__global__ void spatial_avg_pool_kernel(
    const float* __restrict__ patch_tokens,  // [50, 256, 128]
    float*       __restrict__ temporal_tokens) // [50, 128]
{
    int step_idx = blockIdx.x; // 0 ... 49
    int dim_idx  = threadIdx.x; // 0 ... 127

    if (step_idx >= TEMPORAL_STEPS || dim_idx >= EMBED_DIM) return;

    float sum = 0.0f;
    for (int p = 0; p < NUM_PATCHES; ++p) {
        int token_offset = step_idx * NUM_PATCHES + p;
        sum += patch_tokens[token_offset * EMBED_DIM + dim_idx];
    }

    temporal_tokens[step_idx * EMBED_DIM + dim_idx] = sum / static_cast<float>(NUM_PATCHES);
}

// ============================================================================
// 4. TEMPORAL SELF-ATTENTION KERNEL
// ============================================================================
// Computes Q * K^T -> [50, 50] Attention map, Softmax, and Context projection V.
// Map: [50 Temporal Tokens, 128 Features] -> [50 Temporal Tokens, 128 Features]
__global__ void temporal_self_attention_kernel(
    const float* __restrict__ temporal_in,  // [50, 128]
    const float* __restrict__ W_q,           // [128, 128]
    const float* __restrict__ W_k,           // [128, 128]
    const float* __restrict__ W_v,           // [128, 128]
    float*       __restrict__ temporal_out, // [50, 128]
    float*       __restrict__ attn_map_out) // [50, 50] optional output debug
{
    // Shared memory buffers for 50 temporal tokens Q, K, V projections
    __shared__ float Q[TEMPORAL_STEPS][EMBED_DIM];
    __shared__ float K[TEMPORAL_STEPS][EMBED_DIM];
    __shared__ float V[TEMPORAL_STEPS][EMBED_DIM];
    __shared__ float A[TEMPORAL_STEPS][TEMPORAL_STEPS]; // [50, 50] Map

    int step_idx = blockIdx.x; // 0 ... 49
    int dim_idx  = threadIdx.x; // 0 ... 127

    // Phase 1: Compute Linear Projections for Q, K, V
    if (step_idx < TEMPORAL_STEPS && dim_idx < EMBED_DIM) {
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            float x = temporal_in[step_idx * EMBED_DIM + d];
            q_val += x * W_q[d * EMBED_DIM + dim_idx];
            k_val += x * W_k[d * EMBED_DIM + dim_idx];
            v_val += x * W_v[d * EMBED_DIM + dim_idx];
        }
        Q[step_idx][dim_idx] = q_val;
        K[step_idx][dim_idx] = k_val;
        V[step_idx][dim_idx] = v_val;
    }
    __syncthreads();

    // Phase 2: Compute Attention Scores A = (Q * K^T) / sqrt(d_k)
    float scale = 1.0f / sqrtf(static_cast<float>(EMBED_DIM));
    if (step_idx < TEMPORAL_STEPS && dim_idx < TEMPORAL_STEPS) {
        int target_step = dim_idx; // Treat dim_idx as column step index
        float score = 0.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            score += Q[step_idx][d] * K[target_step][d];
        }
        A[step_idx][target_step] = score * scale;
    }
    __syncthreads();

    // Phase 3: Row-wise Softmax Normalization over [50, 50]
    if (dim_idx == 0 && step_idx < TEMPORAL_STEPS) {
        float max_score = A[step_idx][0];
        for (int j = 1; j < TEMPORAL_STEPS; ++j) {
            if (A[step_idx][j] > max_score) max_score = A[step_idx][j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            A[step_idx][j] = expf(A[step_idx][j] - max_score);
            sum_exp += A[step_idx][j];
        }
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            A[step_idx][j] /= sum_exp;
            if (attn_map_out != nullptr) {
                attn_map_out[step_idx * TEMPORAL_STEPS + j] = A[step_idx][j];
            }
        }
    }
    __syncthreads();

    // Phase 4: Context Vector Aggregation A * V
    if (step_idx < TEMPORAL_STEPS && dim_idx < EMBED_DIM) {
        float context = 0.0f;
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            context += A[step_idx][j] * V[j][dim_idx];
        }
        temporal_out[step_idx * EMBED_DIM + dim_idx] = context;
    }
}

// ============================================================================
// 5. TEMPORAL AGGREGATION KERNEL
// ============================================================================
// Aggregates across 50 temporal tokens into 1 pooled sequence vector [128]
__global__ void temporal_avg_pool_kernel(
    const float* __restrict__ temporal_tokens, // [50, 128]
    float*       __restrict__ pooled_seq)       // [128]
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= EMBED_DIM) return;

    float sum = 0.0f;
    for (int t = 0; t < TEMPORAL_STEPS; ++t) {
        sum += temporal_tokens[t * EMBED_DIM + d];
    }
    pooled_seq[d] = sum / static_cast<float>(TEMPORAL_STEPS);
}

// ============================================================================
// 6. LINEAR CLASSIFIER KERNEL
// ============================================================================
// Projects [128] sequence vector into [1000] logits using W_class [128, 1000]
__global__ void linear_classifier_kernel(
    const float* __restrict__ pooled_seq, // [128]
    const float* __restrict__ W_class,    // [128, 1000]
    float*       __restrict__ logits,     // [1000]
    int embed_dim, int num_classes)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_classes) return;

    float score = 0.0f;
    for (int d = 0; d < embed_dim; ++d) {
        score += pooled_seq[d] * W_class[d * num_classes + c];
    }
    logits[c] = score;
}

// ============================================================================
// 7. SOFTMAX LOSS & BACKWARD GRADIENT PASS
// ============================================================================
__global__ void softmax_cross_entropy_kernel(
    const float* __restrict__ logits,      // [1000]
    const float* __restrict__ pooled_seq,  // [128]
    const int*   __restrict__ label,
    const float* __restrict__ W_class,     // [128, 1000]
    float*       __restrict__ dL_dpooled,  // [128]
    float*       __restrict__ dL_dW_class, // [128, 1000]
    float*       __restrict__ loss_out,
    int*         __restrict__ correct_out,
    int embed_dim, int num_classes)
{
    int target = *label;

    float max_logit = logits[0];
    int argmax = 0;
    for (int c = 1; c < num_classes; ++c) {
        if (logits[c] > max_logit) {
            max_logit = logits[c];
            argmax = c;
        }
    }

    if (threadIdx.x == 0) {
        *correct_out = (argmax == target) ? 1 : 0;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        sum_exp += expf(logits[c] - max_logit);
    }

    float prob_target = expf(logits[target] - max_logit) / sum_exp;
    if (threadIdx.x == 0) {
        float raw_loss = -logf(fminf(fmaxf(prob_target, 1e-7f), 1.0f - 1e-7f));
        *loss_out = (raw_loss < 1e-7f) ? 0.0f : raw_loss;
    }

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < embed_dim) {
        float pooled_val = pooled_seq[d];
        float grad_p = 0.0f;

        for (int c = 0; c < num_classes; ++c) {
            float p_c = expf(logits[c] - max_logit) / sum_exp;
            float dL_dlogit = p_c - (c == target ? 1.0f : 0.0f);
            grad_p += dL_dlogit * W_class[d * num_classes + c];
            atomicAdd(&dL_dW_class[d * num_classes + c], dL_dlogit * pooled_val);
        }
        dL_dpooled[d] = grad_p;
    }
}

// Un-pools loss gradients back to patch tokens across all 50 x 256 spatio-temporal tokens
__global__ void spatiotemporal_backward_kernel(
    const float* __restrict__ dL_dpooled,  // [128]
    const float* __restrict__ densities,   // [50 * 256, 2048]
    float*       __restrict__ dL_dW_proj)   // [2048, 128]
{
    int w_idx   = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (w_idx >= WORDS_PER_PATCH || dim_idx >= EMBED_DIM) return;

    float grad_acc = 0.0f;
    float dL_dt = dL_dpooled[dim_idx] / static_cast<float>(TEMPORAL_STEPS * NUM_PATCHES);

    for (int tok = 0; tok < TEMPORAL_STEPS * NUM_PATCHES; ++tok) {
        float density = densities[tok * WORDS_PER_PATCH + w_idx];
        grad_acc += density * dL_dt;
    }

    dL_dW_proj[w_idx * EMBED_DIM + dim_idx] = grad_acc;
}

// ============================================================================
// 8. ADAMW OPTIMIZER KERNEL
// ============================================================================
__global__ void adamw_update_kernel(
    float* __restrict__ weights, const float* __restrict__ grad,
    float* __restrict__ m, float* __restrict__ v,
    int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx] + weight_decay * weights[idx];
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * (g * g);

    float m_hat = m[idx] / (1.0f - powf(beta1, static_cast<float>(step)));
    float v_hat = v[idx] / (1.0f - powf(beta2, static_cast<float>(step)));

    weights[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ============================================================================
// 9. MAIN PIPELINE EXECUTION
// ============================================================================
int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Complete Spatio-Temporal Hypercube Pipeline                         " << std::endl;
    std::cout << "======================================================================" << std::endl;

    // 1. Raw Simulation Buffer Initialization
    bool* d_raw_hypercube = nullptr;
    size_t raw_bytes = static_cast<size_t>(RAW_FRAMES) * NUM_PATCHES * WORDS_PER_FRAME * 32 * sizeof(bool);
    CUDA_CHECK(cudaMalloc(&d_raw_hypercube, raw_bytes));

    int h_target_label = 42;
    int* d_target_label = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target_label, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_target_label, &h_target_label, sizeof(int), cudaMemcpyHostToDevice));

    // 2. Weights Allocations
    int num_proj_weights  = WORDS_PER_PATCH * EMBED_DIM; // 2048 * 128
    int num_class_weights = EMBED_DIM * NUM_CLASSES;      // 128 * 1000
    int num_attn_weights  = EMBED_DIM * EMBED_DIM;        // 128 * 128

    float *d_W_proj = nullptr, *d_dL_dW_proj = nullptr, *d_m_proj = nullptr, *d_v_proj = nullptr;
    float *d_W_class = nullptr, *d_dL_dW_class = nullptr, *d_m_class = nullptr, *d_v_class = nullptr;
    float *d_W_q = nullptr, *d_W_k = nullptr, *d_W_v = nullptr;

    CUDA_CHECK(cudaMalloc(&d_W_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_proj, num_proj_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_class, num_class_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_q, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_k, num_attn_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_v, num_attn_weights * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_proj, 0, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_proj, 0, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_class, 0, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_class, 0, num_class_weights * sizeof(float)));

    // Xavier initialization
    std::vector<float> h_W_proj(num_proj_weights), h_W_class(num_class_weights), h_W_attn(num_attn_weights);
    for (int i = 0; i < num_proj_weights; ++i) h_W_proj[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < num_class_weights; ++i) h_W_class[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < num_attn_weights; ++i) h_W_attn[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.02f;

    CUDA_CHECK(cudaMemcpy(d_W_proj, h_W_proj.data(), num_proj_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_class, h_W_class.data(), num_class_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_q, h_W_attn.data(), num_attn_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_k, h_W_attn.data(), num_attn_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_v, h_W_attn.data(), num_attn_weights * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Intermediate Execution Buffers
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

    std::cout << "[+] System Initialized to Data Flow Specs:" << std::endl;
    std::cout << "    - Raw Buffer Size: " << (raw_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "    - Tubelets (T): " << TEMPORAL_STEPS << " steps x " << NUM_PATCHES << " patches" << std::endl;
    std::cout << "    - Words per Tubelet Patch: " << WORDS_PER_PATCH << std::endl;
    std::cout << "    - W_proj Shape: [" << WORDS_PER_PATCH << ", " << EMBED_DIM << "]" << std::endl;
    std::cout << "    - Self-Attention Map Shape: [" << TEMPORAL_STEPS << ", " << TEMPORAL_STEPS << "]" << std::endl;
    std::cout << "    - W_class Shape: [" << EMBED_DIM << ", " << NUM_CLASSES << "]" << std::endl;

    // Grid Dimensions
    dim3 fwd_proj_grid(NUM_PATCHES, TEMPORAL_STEPS);
    dim3 fwd_proj_block(EMBED_DIM);

    int total_raw_bools = RAW_FRAMES * NUM_PATCHES * WORDS_PER_FRAME * 32;
    int mutate_blocks = (total_raw_bools + 255) / 256;

    int opt_proj_blocks  = (num_proj_weights + 255) / 256;
    int opt_class_blocks = (num_class_weights + 255) / 256;

    // Training Loop Execution
    for (int step = 1; step <= 10; ++step) {
        // Step 0: Raw Simulation Mutation
        mutate_raw_hypercube_kernel<<<mutate_blocks, 256>>>(d_raw_hypercube, 1337 + step);

        // Step 1: Spatiotemporal Tubelet Projection
        spatiotemporal_projection_kernel<<<fwd_proj_grid, fwd_proj_block>>>(
            d_raw_hypercube, d_W_proj, d_patch_tokens, d_densities
        );

        // Step 2: Spatial Aggregation across N=256 patches -> [50, 128]
        spatial_avg_pool_kernel<<<TEMPORAL_STEPS, EMBED_DIM>>>(
            d_patch_tokens, d_temporal_tokens
        );

        // Step 3: Temporal Self-Attention (Q * K^T -> [50, 50] map -> Context V)
        temporal_self_attention_kernel<<<TEMPORAL_STEPS, EMBED_DIM>>>(
            d_temporal_tokens, d_W_q, d_W_k, d_W_v, d_attn_temporal_out, d_attn_map
        );

        // Step 4: Temporal Aggregation across T=50 steps -> [128]
        temporal_avg_pool_kernel<<<(EMBED_DIM + 255) / 256, 256>>>(
            d_attn_temporal_out, d_pooled_seq
        );

        // Step 5: Linear Classifier -> [1000]
        linear_classifier_kernel<<<(NUM_CLASSES + 255) / 256, 256>>>(
            d_pooled_seq, d_W_class, d_logits, EMBED_DIM, NUM_CLASSES
        );

        // Step 6: Softmax Loss & Backward Pass
        CUDA_CHECK(cudaMemset(d_dL_dW_class, 0, num_class_weights * sizeof(float)));
        softmax_cross_entropy_kernel<<<(EMBED_DIM + 255) / 256, 256>>>(
            d_logits, d_pooled_seq, d_target_label, d_W_class,
            d_dL_dpooled, d_dL_dW_class, d_loss_out, d_correct_out,
            EMBED_DIM, NUM_CLASSES
        );

        spatiotemporal_backward_kernel<<<WORDS_PER_PATCH, EMBED_DIM>>>(
            d_dL_dpooled, d_densities, d_dL_dW_proj
        );

        // Step 7: AdamW Parameter Updates
        adamw_update_kernel<<<opt_proj_blocks, 256>>>(
            d_W_proj, d_dL_dW_proj, d_m_proj, d_v_proj,
            num_proj_weights, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_class_blocks, 256>>>(
            d_W_class, d_dL_dW_class, d_m_class, d_v_class,
            num_class_weights, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        float h_loss = 0.0f;
        int h_correct = 0;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_out, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_correct, d_correct_out, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "[Step " << std::setw(2) << step << "/10] "
                  << "Loss: " << std::fixed << std::setprecision(5) << h_loss
                  << " | Accuracy: " << std::setprecision(1) << (static_cast<float>(h_correct) * 100.0f) << "%"
                  << " (Target Class: " << h_target_label << ")"
                  << std::endl;
    }

    // Cleanup Resources
    CUDA_CHECK(cudaFree(d_raw_hypercube)); CUDA_CHECK(cudaFree(d_target_label));
    CUDA_CHECK(cudaFree(d_W_proj)); CUDA_CHECK(cudaFree(d_dL_dW_proj));
    CUDA_CHECK(cudaFree(d_m_proj)); CUDA_CHECK(cudaFree(d_v_proj));
    CUDA_CHECK(cudaFree(d_W_class)); CUDA_CHECK(cudaFree(d_dL_dW_class));
    CUDA_CHECK(cudaFree(d_m_class)); CUDA_CHECK(cudaFree(d_v_class));
    CUDA_CHECK(cudaFree(d_W_q)); CUDA_CHECK(cudaFree(d_W_k)); CUDA_CHECK(cudaFree(d_W_v));
    CUDA_CHECK(cudaFree(d_patch_tokens)); CUDA_CHECK(cudaFree(d_densities));
    CUDA_CHECK(cudaFree(d_temporal_tokens)); CUDA_CHECK(cudaFree(d_attn_temporal_out));
    CUDA_CHECK(cudaFree(d_attn_map)); CUDA_CHECK(cudaFree(d_pooled_seq));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_dL_dpooled));
    CUDA_CHECK(cudaFree(d_loss_out)); CUDA_CHECK(cudaFree(d_correct_out));

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  Spatio-Temporal Pipeline Execution Complete!                        " << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}