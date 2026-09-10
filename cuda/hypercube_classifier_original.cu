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
// Total booleans in our hypercube manifold (4 MB bitfield = 33,554,432 bits)
// Represented in GPU memory as 4,194,304 contiguous bools (1 byte per bool for fast execution)
constexpr int HYPERCUBE_BOOLS = 4194304; 

// Vision Transformer spatial tokenization parameters
constexpr int NUM_PATCHES     = 256;     // Number of spatial patches/tokens per timeframe snapshot
constexpr int NUM_TIMEFRAMES  = 4;       // Number of temporal snapshot states (T)
constexpr int TOTAL_TOKENS    = NUM_PATCHES * NUM_TIMEFRAMES; // Total spatio-temporal tokens (T * N)

constexpr int BOOLS_PER_PATCH = HYPERCUBE_BOOLS / NUM_PATCHES; // 16,384 bools / patch
constexpr int WORDS_PER_PATCH = BOOLS_PER_PATCH / 32;          // 512 logical 32-bit word equivalents
constexpr int TOTAL_WORDS     = WORDS_PER_PATCH * NUM_PATCHES; // 131072 words in this hypercube
constexpr int EMBED_DIM       = 128;                           // Transformer embedding dimension (D)
constexpr int NUM_CLASSES     = 10;                            // Downstream classification categories

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
// 1. MUTATING ENGINE KERNEL (Multi-Timeframe Sequence Generation)
// ============================================================================
// Simulates continuous dynamic state transitions across time steps (t = 0 ... T-1).
// Frame 0 holds the initial manifold baseline; subsequent frames apply high-throughput
// XOR-shift pseudo-random bit mutators to model temporal hypercube evolution.
__global__ void mutate_hypercube_sequence_kernel(
    bool* __restrict__ sequence_hypercube, // [NUM_TIMEFRAMES, HYPERCUBE_BOOLS]
    uint32_t seed) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= HYPERCUBE_BOOLS) return;

    // Load baseline state at t = 0
    bool current_state = sequence_hypercube[tid]; 

    // Mutate across temporal snapshots and write into multi-frame buffer
    for (int t = 0; t < NUM_TIMEFRAMES; ++t) {
        if (t > 0) {
            // High-throughput 32-bit XOR-shift mutator per bit position
            uint32_t x = tid ^ (seed + t * 0x9e3779b9);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // 50% probability flip on bit evolution
            current_state = (x & 0x1) ? !current_state : current_state;
        }
        sequence_hypercube[t * HYPERCUBE_BOOLS + tid] = current_state;
    }
}

// ============================================================================
// 2. SPATIO-TEMPORAL FORWARD EXTRACTION KERNEL (nvJPEG-style Vectorized Loads)
// ============================================================================
// Reads sub-byte/boolean states directly using 32-bit (uchar4) vectorized memory 
// instructions. Parallelized across both Spatial Patches (blockIdx.x) and Temporal 
// Timeframes (blockIdx.y), projecting boolean density slices into continuous embeddings.
__global__ void vit_embed_forward_temporal_kernel(
    const bool* __restrict__ sequence_state, // [NUM_TIMEFRAMES, HYPERCUBE_BOOLS], not a good name!
    const float* __restrict__ W_proj,         // [WORDS_PER_PATCH, EMBED_DIM]
    float*       __restrict__ tokens,         // [NUM_TIMEFRAMES, NUM_PATCHES, EMBED_DIM]
    float*       __restrict__ densities)      // [NUM_TIMEFRAMES, NUM_PATCHES, WORDS_PER_PATCH]
{
    int t_idx     = blockIdx.y; // Temporal timeframe index (0 ... T-1)
    int patch_idx = blockIdx.x; // Spatial patch index (0 ... N-1)
    int dim_idx   = threadIdx.x; // Embedding dimension channel (0 ... D-1)

    if (t_idx >= NUM_TIMEFRAMES || patch_idx >= NUM_PATCHES || dim_idx >= EMBED_DIM) return;

    int token_offset = (t_idx * NUM_PATCHES + patch_idx);
    const bool* current_patch_state = sequence_state + t_idx * HYPERCUBE_BOOLS + patch_idx * BOOLS_PER_PATCH;

    // Treat boolean region as contiguous array of uchar4 (4-byte vectorized access)
    const uchar4* state_vec = reinterpret_cast<const uchar4*>(current_patch_state);

    float total_projection = 0.0f;
    int num_vecs = BOOLS_PER_PATCH / 4; // 4096 uchar4 vectors per patch, each uchar4 has 4 bools.

    for (int i = 0; i < num_vecs; ++i) {
        // Vectorized load of 4 boolean values simultaneously
        uchar4 b4 = state_vec[i];

        // Convert boolean states directly to float densities (1.0f or 0.0f)
        float v0 = b4.x ? 1.0f : 0.0f;
        float v1 = b4.y ? 1.0f : 0.0f;
        float v2 = b4.z ? 1.0f : 0.0f;
        float v3 = b4.w ? 1.0f : 0.0f;

        // Group into 32-bool density word equivalents for backward compatibility
        // Map to logical word equivalent (8 x uchar4 = 32 bools = 1 word)
        int word_idx = i / 8; // 8 x 4-bools = 32 bools = 1 word

        // Density Cache Phase: thread 0 records word-level active bit densities
        // required during backpropagation for exact projection gradient calculation
        if (densities != nullptr && dim_idx == 0 && (i % 8 == 0)) {
            // Aggregate density across 32 boolean elements
            float density_sum = 0.0f;
            for (int k = 0; k < 8; ++k) {
                uchar4 sub_b4 = state_vec[i + k];
                density_sum += (sub_b4.x ? 1.0f : 0.0f) + (sub_b4.y ? 1.0f : 0.0f) +
                               (sub_b4.z ? 1.0f : 0.0f) + (sub_b4.w ? 1.0f : 0.0f);
            }
            // Store normalized activation density [0.0, 1.0] for this word slice
            densities[token_offset * WORDS_PER_PATCH + word_idx] = density_sum / 32.0f;
        }

        // Multiply activation density by projection weights W_proj[word_idx, dim_idx]
        // Linear embedding projection sum
        // W_proj dimension shape: [WORDS_PER_PATCH, EMBED_DIM]
        // Since each 32-bit word contain 4 uchar4, the w_sub stays same for 4 following uchar4.
        // The total_projection is accumulated correctly via +=.
        int w_sub = word_idx;
        total_projection += (v0 + v1 + v2 + v3) * 0.25f * W_proj[w_sub * EMBED_DIM + dim_idx];
    }

    // Write final spatio-temporal token embedding output
    // tokens dimension shape: [NUM_PATCHES, EMBED_DIM]
    tokens[token_offset * EMBED_DIM + dim_idx] = total_projection;
}

// ============================================================================
// 3. SPATIO-TEMPORAL AVERAGE POOLING & CLASSIFICATION
// ============================================================================
// Global Spatio-Temporal Average Pooling: Aggregates across all (T * N) tokens
// into a unified D-dimensional sequence embedding vector.
// This final compression is achived via averaging.
// For each dimension, the value is averaged via the same dimension value across all num_patches.
__global__ void global_spatiotemporal_pool_kernel(
    const float* __restrict__ tokens, // [TOTAL_TOKENS, EMBED_DIM]
    float*       __restrict__ pooled, // [EMBED_DIM]
    int total_tokens, int embed_dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= embed_dim) return;

    float sum = 0.0f;
    for (int tok = 0; tok < total_tokens; ++tok) {
        sum += tokens[tok * embed_dim + d];
    }
    pooled[d] = sum / static_cast<float>(total_tokens);
}

// This functin takes the hypercube snapshot summary vector (embed_dim)
// and computes logits for each category.
__global__ void linear_classifier_kernel(
    const float* __restrict__ pooled,  // [EMBED_DIM], the feature values are aggregated into pooled from [TOTAL_TOKENS, EMBED_DIM] via averaging.
    const float* __restrict__ W_class, // [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ logits,  // [NUM_CLASSES]
    int embed_dim, int num_classes)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_classes) return;

    float score = 0.0f;
    for (int d = 0; d < embed_dim; ++d) {
        // [EMBED_DIM] * [EMBED_DIM, NUM_CLASSES] = [NUM_CLASSES]
        score += pooled[d] * W_class[d * num_classes + c];
    }
    logits[c] = score;
}

// ============================================================================
// 4. SOFTMAX LOSS & SPATIO-TEMPORAL BACKPROPAGATION
// ============================================================================
// Evaluates Softmax Cross-Entropy loss, computes classifier weight gradients,
// and un-pools loss gradients back to all (T * N) spatio-temporal tokens.
__global__ void softmax_cross_entropy_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ pooled,     // Forward activations needed for dL/dW_class
    const int*   __restrict__ label,
    const float* __restrict__ W_class,    // Shape: [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ dL_dtokens, // Shape: [TOTAL_TOKENS, EMBED_DIM]
    float*       __restrict__ dL_dW_class,
    float*       __restrict__ loss_out,
    int*         __restrict__ correct_out,
    int total_tokens, int embed_dim, int num_classes)
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

    // Softmax Denominator
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        sum_exp += expf(logits[c] - max_logit);
    }

    // Cross-Entropy Loss Calculation
    // Clamp probability to strictly avoid log(1.0f) signed zero artifacts
    float prob_target = expf(logits[target] - max_logit) / sum_exp;
    if (threadIdx.x == 0) {
        float raw_loss = -logf(fminf(fmaxf(prob_target, 1e-7f), 1.0f - 1e-7f));
        *loss_out = (raw_loss < 1e-7f) ? 0.0f : raw_loss;
    }

    // Gradient Backpropagation
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < embed_dim) {
        float pooled_val = pooled[d];  // Read activation for feature dimension d
        float grad_pooled = 0.0f;

        for (int c = 0; c < num_classes; ++c) {
            float p_c = expf(logits[c] - max_logit) / sum_exp;
            float dL_dlogit = p_c - (c == target ? 1.0f : 0.0f);

            // 1. Backprop into pooled representations: dL/dpooled[d] = sum_c (dL/dz_c * W_class[d, c])
            grad_pooled += dL_dlogit * W_class[d * num_classes + c];
            // 2. Weight gradient accumulation: dL/dW_class[d, c] = dL/dz_c * pooled[d]
            // dL_dW_class is the gradient of the loss function L with respect to the class representations matrix.
            // Shape is [EMBED_DIM, NUM_CLASSES]
            // W_class is used to convert snapshot vector [EMBED_DIM] into class vector/logits [NUM_CLASSES].
            // "* num_classes" refers to the second value (number of columns) in shape.
            atomicAdd(&dL_dW_class[d * num_classes + c], dL_dlogit * pooled_val);
        }

        // Divide gradient evenly across spatial patches for un-pooling
        // dL_dtokens is the gradient of the loss function L with respect to the patch token representations matrix (tokens).
        // Shape [TOTAL_TOKENS, EMBED_DIM]
        float grad_token = grad_pooled / static_cast<float>(total_tokens);
        for (int tok = 0; tok < total_tokens; ++tok) {
            dL_dtokens[tok * embed_dim + d] = grad_token;
        }
    }
}

// ============================================================================
// 5. TEMPORAL BACKWARD GRADIENT KERNEL
// ============================================================================
// Computes analytical gradients dL/dW_proj w.r.t projection weights by integrating 
// spatial densities and loss gradients across all timeframes (t = 0 ... T-1).
__global__ void vit_embed_backward_temporal_kernel(
    const float* __restrict__ dL_dtokens, // [TOTAL_TOKENS, EMBED_DIM]
    const float* __restrict__ densities,  // [TOTAL_TOKENS, WORDS_PER_PATCH]
    float*       __restrict__ dL_dW_proj)  // [WORDS_PER_PATCH, EMBED_DIM]
{
    int w_idx   = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (w_idx >= WORDS_PER_PATCH || dim_idx >= EMBED_DIM) return;

    // Accumulate projection gradients across every spatio-temporal token
    float grad_acc = 0.0f;
    for (int tok = 0; tok < TOTAL_TOKENS; ++tok) {
        float density = densities[tok * WORDS_PER_PATCH + w_idx];
        float dL_dt   = dL_dtokens[tok * EMBED_DIM + dim_idx];
        grad_acc += density * dL_dt;
    }

    dL_dW_proj[w_idx * EMBED_DIM + dim_idx] = grad_acc;
}

// ============================================================================
// 6. ADAMW OPTIMIZER KERNEL
// ============================================================================
// Fused CUDA AdamW parameter update kernel supporting L2 weight decay.
__global__ void adamw_update_kernel(
    float* __restrict__ weights,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int size, float lr, float beta1, float beta2, float eps, float weight_decay, int step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Decoupled Weight Decay
    float g = grad[idx] + weight_decay * weights[idx];

    // First and second moment updates
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * (g * g);

    // Bias corrections
    float m_hat = m[idx] / (1.0f - powf(beta1, static_cast<float>(step)));
    float v_hat = v[idx] / (1.0f - powf(beta2, static_cast<float>(step)));

    // Parameter update
    weights[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ============================================================================
// 7. MAIN PIPELINE EXECUTION
// ============================================================================
int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Multi-Timeframe Spatio-Temporal Boolean Hypercube Classifier         " << std::endl;
    std::cout << "======================================================================" << std::endl;

    // 1. Allocate Multi-Timeframe Sequence Hypercube Buffer
    bool* d_sequence_hypercube = nullptr;
    size_t seq_bytes = static_cast<size_t>(NUM_TIMEFRAMES) * HYPERCUBE_BOOLS * sizeof(bool);
    CUDA_CHECK(cudaMalloc(&d_sequence_hypercube, seq_bytes));
    CUDA_CHECK(cudaMemset(d_sequence_hypercube, 1, seq_bytes)); // Baseline initialization

    int h_target_label = 3;
    int* d_target_label = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target_label, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_target_label, &h_target_label, sizeof(int), cudaMemcpyHostToDevice));

    // 2. Projection & Classifier Weights
    int num_proj_weights  = WORDS_PER_PATCH * EMBED_DIM;
    int num_class_weights = EMBED_DIM * NUM_CLASSES;

    float *d_W_proj = nullptr, *d_dL_dW_proj = nullptr, *d_m_proj = nullptr, *d_v_proj = nullptr;
    float *d_W_class = nullptr, *d_dL_dW_class = nullptr, *d_m_class = nullptr, *d_v_class = nullptr;

    CUDA_CHECK(cudaMalloc(&d_W_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_proj, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_proj, num_proj_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_W_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dW_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_class, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_class, num_class_weights * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_proj, 0, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_proj, 0, num_proj_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_class, 0, num_class_weights * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_class, 0, num_class_weights * sizeof(float)));

    // Xavier/Glorot uniform initialization on host
    std::vector<float> h_W_proj_init(num_proj_weights);
    std::vector<float> h_W_class_init(num_class_weights);
    for (int i = 0; i < num_proj_weights; ++i) h_W_proj_init[i] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f;
    for (int i = 0; i < num_class_weights; ++i) h_W_class_init[i] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f;

    CUDA_CHECK(cudaMemcpy(d_W_proj, h_W_proj_init.data(), num_proj_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_class, h_W_class_init.data(), num_class_weights * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Intermediate Spatio-Temporal Buffers
    float *d_tokens = nullptr, *d_densities = nullptr, *d_pooled = nullptr;
    float *d_logits = nullptr, *d_dL_dtokens = nullptr;
    float *d_loss_out = nullptr;
    int*   d_correct_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tokens, TOTAL_TOKENS * EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_densities, TOTAL_TOKENS * WORDS_PER_PATCH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pooled, EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dtokens, TOTAL_TOKENS * EMBED_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_loss_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct_out, sizeof(int)));

    // Concurrent CUDA Streams setup
    cudaStream_t stream_mutate, stream_train;
    CUDA_CHECK(cudaStreamCreate(&stream_mutate));
    CUDA_CHECK(cudaStreamCreate(&stream_train));

    std::cout << "[+] System Initialized." << std::endl;
    std::cout << "    - Timeframes (T): " << NUM_TIMEFRAMES << std::endl;
    std::cout << "    - Spatial Patches per Frame: " << NUM_PATCHES << std::endl;
    std::cout << "    - Total Spatio-Temporal Tokens: " << TOTAL_TOKENS << std::endl;

    // Grid Dimensions mapping
    dim3 fwd_grid(NUM_PATCHES, NUM_TIMEFRAMES); // X = Spatial Patches, Y = Temporal Snapshots
    dim3 fwd_block(EMBED_DIM);
    dim3 bwd_grid(WORDS_PER_PATCH);
    dim3 bwd_block(EMBED_DIM);

    int mutate_threads = 256;
    int mutate_blocks  = (HYPERCUBE_BOOLS + mutate_threads - 1) / mutate_threads;

    int opt_threads = 256;
    int opt_proj_blocks  = (num_proj_weights + opt_threads - 1) / opt_threads;
    int opt_class_blocks = (num_class_weights + opt_threads - 1) / opt_threads;

    // Training Loop Execution
    for (int step = 1; step <= 10; ++step) {
        // Step 1: Mutate Sequence States across time steps t = 0 ... T-1
        mutate_hypercube_sequence_kernel<<<mutate_blocks, mutate_threads, 0, stream_mutate>>>(
            d_sequence_hypercube, 1337 + step
        );

        // Step 2: Extract Spatio-Temporal Token Embeddings via Vectorized GPU Loads
        vit_embed_forward_temporal_kernel<<<fwd_grid, fwd_block, 0, stream_train>>>(
            d_sequence_hypercube, d_W_proj, d_tokens, d_densities
        );

        // Step 3: Global Spatio-Temporal Pooling across [TOTAL_TOKENS, EMBED_DIM]
        int pool_blocks = (EMBED_DIM + 255) / 256;
        global_spatiotemporal_pool_kernel<<<pool_blocks, 256, 0, stream_train>>>(
            d_tokens, d_pooled, TOTAL_TOKENS, EMBED_DIM
        );

        // Step 4: Downstream Classification
        int class_blocks = (NUM_CLASSES + 255) / 256;
        linear_classifier_kernel<<<class_blocks, 256, 0, stream_train>>>(
            d_pooled, d_W_class, d_logits, EMBED_DIM, NUM_CLASSES
        );

        // Step 5: Softmax Cross-Entropy Loss & Un-pooling Gradient Calculation
        CUDA_CHECK(cudaMemsetAsync(d_dL_dW_class, 0, num_class_weights * sizeof(float), stream_train));
        softmax_cross_entropy_kernel<<<pool_blocks, 256, 0, stream_train>>>(
            d_logits, d_pooled, d_target_label, d_W_class, d_dL_dtokens, d_dL_dW_class,
            d_loss_out, d_correct_out, TOTAL_TOKENS, EMBED_DIM, NUM_CLASSES
        );

        // Step 6: Backward Pass across Spatio-Temporal Token Densities
        vit_embed_backward_temporal_kernel<<<bwd_grid, bwd_block, 0, stream_train>>>(
            d_dL_dtokens, d_densities, d_dL_dW_proj
        );

        // Step 7: Fused AdamW Optimizer Parameter Updates
        adamw_update_kernel<<<opt_proj_blocks, opt_threads, 0, stream_train>>>(
            d_W_proj, d_dL_dW_proj, d_m_proj, d_v_proj,
            num_proj_weights, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_class_blocks, opt_threads, 0, stream_train>>>(
            d_W_class, d_dL_dW_class, d_m_class, d_v_class,
            num_class_weights, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        // Asynchronous readout and synchronization
        float h_loss = 0.0f;
        int h_correct = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_loss_out, sizeof(float), cudaMemcpyDeviceToHost, stream_train));
        CUDA_CHECK(cudaMemcpyAsync(&h_correct, d_correct_out, sizeof(int), cudaMemcpyDeviceToHost, stream_train));
        CUDA_CHECK(cudaStreamSynchronize(stream_train));

        std::cout << "[Step " << std::setw(2) << step << "/10] "
                  << "Loss: " << std::fixed << std::setprecision(5) << h_loss
                  << " | Accuracy: " << std::setprecision(1) << (static_cast<float>(h_correct) * 100.0f) << "%"
                  << " (Target Class: " << h_target_label << ")"
                  << std::endl;
    }

    // Free device resources & destroy streams
    CUDA_CHECK(cudaStreamDestroy(stream_mutate));
    CUDA_CHECK(cudaStreamDestroy(stream_train));

    CUDA_CHECK(cudaFree(d_sequence_hypercube));
    CUDA_CHECK(cudaFree(d_target_label));
    CUDA_CHECK(cudaFree(d_W_proj)); CUDA_CHECK(cudaFree(d_dL_dW_proj));
    CUDA_CHECK(cudaFree(d_m_proj)); CUDA_CHECK(cudaFree(d_v_proj));
    CUDA_CHECK(cudaFree(d_W_class)); CUDA_CHECK(cudaFree(d_dL_dW_class));
    CUDA_CHECK(cudaFree(d_m_class)); CUDA_CHECK(cudaFree(d_v_class));
    CUDA_CHECK(cudaFree(d_tokens)); CUDA_CHECK(cudaFree(d_densities));
    CUDA_CHECK(cudaFree(d_pooled)); CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_dL_dtokens)); CUDA_CHECK(cudaFree(d_loss_out));
    CUDA_CHECK(cudaFree(d_correct_out));

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  Multi-Timeframe Execution Complete!                                 " << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}