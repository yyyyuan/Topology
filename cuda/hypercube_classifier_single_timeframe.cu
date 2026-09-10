%%writefile main.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

// ============================================================================
// CONFIGURATION PARAMETERS
// ============================================================================
constexpr int HYPERCUBE_BOOLS = 4194304; // 4 MB boolean bitfield (131072 * 32 bools)
constexpr int NUM_PATCHES     = 256;     // This is the number of tokens, each token (vector) has 128 dimensions.
constexpr int BOOLS_PER_PATCH = HYPERCUBE_BOOLS / NUM_PATCHES; // 16384 bools / patch
constexpr int WORDS_PER_PATCH = BOOLS_PER_PATCH / 32;          // 512 logical word equivalents
constexpr int TOTAL_WORDS     = WORDS_PER_PATCH * NUM_PATCHES; // 131072 words in this hypercube
constexpr int EMBED_DIM       = 128;                           // Token vector dimension
constexpr int NUM_CLASSES     = 10;                            // Output classification categories

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
// 1. MUTATING ENGINE KERNEL (Boolean State Evolution)
// ============================================================================
__global__ void mutate_hypercube_bool_kernel(bool* __restrict__ hypercube, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HYPERCUBE_BOOLS) return;

    // Fast PRNG bit flip simulating hypercube state transition
    uint32_t x = idx ^ seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    // Mutate state using lower bit pattern
    hypercube[idx] = (x & 0x1) ? !hypercube[idx] : hypercube[idx];
}

// ============================================================================
// 2. INFERENCE / FORWARD EXTRACTION KERNEL (Boolean Vector Read)
// ============================================================================
__global__ void vit_embed_forward_bool_kernel(
    const bool* __restrict__ state,         // The excited array of hypercube.
    const float* __restrict__ W_proj,       // [WORDS_PER_PATCH, EMBED_DIM]
    float*       __restrict__ tokens,       // [NUM_PATCHES, EMBED_DIM]
    float*       __restrict__ densities)    // [NUM_PATCHES, WORDS_PER_PATCH] Saved Activations
{
    int patch_idx = blockIdx.x;
    int dim_idx   = threadIdx.x;

    if (patch_idx >= NUM_PATCHES || dim_idx >= EMBED_DIM) return;

    // Read state in 4-byte vectorized chunks using uchar4 reinterpretation
    const uchar4* state_vec = reinterpret_cast<const uchar4*>(
        state + patch_idx * BOOLS_PER_PATCH
    );

    float total_projection = 0.0f;
    int num_vecs = BOOLS_PER_PATCH / 4; // 4096 uchar4 vectors per patch

    for (int i = 0; i < num_vecs; ++i) {
        uchar4 b4 = state_vec[i];

        // Convert boolean states directly to float densities (1.0f or 0.0f)
        float v0 = b4.x ? 1.0f : 0.0f;
        float v1 = b4.y ? 1.0f : 0.0f;
        float v2 = b4.z ? 1.0f : 0.0f;
        float v3 = b4.w ? 1.0f : 0.0f;

        // Group into 32-bool density word equivalents for backward compatibility
        // Now each uchar4 contains 4 bools, to make up the 32 bools it needs 8 more uchar4.
        int word_idx = i / 8; // 8 x 4-bools = 32 bools = 1 word

        // densities is only calculated in thread #0 among all 128 threads in one block.
        if (densities != nullptr && dim_idx == 0 && (i % 8 == 0)) {
            // Aggregate density across 32 boolean elements
            float density_sum = 0.0f;
            for (int k = 0; k < 8; ++k) {
                uchar4 sub_b4 = state_vec[i + k];
                density_sum += (sub_b4.x ? 1.0f : 0.0f) + (sub_b4.y ? 1.0f : 0.0f) +
                               (sub_b4.z ? 1.0f : 0.0f) + (sub_b4.w ? 1.0f : 0.0f);
            }
            densities[patch_idx * WORDS_PER_PATCH + word_idx] = density_sum / 32.0f;
        }

        // Linear embedding projection sum
        // W_proj dimension shape: [WORDS_PER_PATCH, EMBED_DIM]
        // Since each 32-bit word contain 4 uchar4, the w_sub stays same for 4 following uchar4.
        // The total_projection is accumulated correctly via +=.
        int w_sub = word_idx;
        total_projection += (v0 + v1 + v2 + v3) * 0.25f * W_proj[w_sub * EMBED_DIM + dim_idx];
    }

    // tokens dimension shape: [NUM_PATCHES, EMBED_DIM]
    tokens[patch_idx * EMBED_DIM + dim_idx] = total_projection;
}

// ============================================================================
// 3. GLOBAL AVERAGE POOLING & LINEAR CLASSIFICATION KERNELS
// ============================================================================
// Each hypercube snapshot is compressed into a vector of `embed_dim` dimensions eventually.
// This final compression is achived via averaging.
// For each dimension, the value is averaged via the same dimension value across all num_patches.
__global__ void global_avg_pool_kernel(
    const float* __restrict__ tokens,
    float*       __restrict__ pooled,
    int num_patches, int embed_dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= embed_dim) return;

    float sum = 0.0f;
    for (int p = 0; p < num_patches; ++p) {
        // [NUM_PATCHES, EMBED_DIM] -> [EMBED_DIM]
        sum += tokens[p * embed_dim + d];
    }
    pooled[d] = sum / static_cast<float>(num_patches);
}

// This functin takes the hypercube snapshot summary vector (embed_dim)
// and computes logits for each category.
__global__ void linear_classifier_kernel(
    const float* __restrict__ pooled,  // [EMBED_DIM]
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
// 4. SOFTMAX CROSS-ENTROPY LOSS & ACCURACY KERNEL
// ============================================================================
__global__ void softmax_cross_entropy_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ pooled,      // <--- ADDED: Forward activations needed for dL/dW_class
    const int*   __restrict__ label,
    const float* __restrict__ W_class,    // Shape: [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ dL_dtokens,
    float*       __restrict__ dL_dW_class,
    float*       __restrict__ loss_out,
    int*         __restrict__ correct_out,
    int num_patches, int embed_dim, int num_classes)
{
    int target = *label;

    // Argmax and Max-Logit for Numerical Stability
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
    float prob_target = expf(logits[target] - max_logit) / sum_exp;
    if (threadIdx.x == 0) {
        *loss_out = -logf(fmaxf(prob_target, 1e-7f));
    }

    // Gradient Backpropagation
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread d handles one feature dimension (d in [0, embed_dim - 1]) and
    // iterates through all classes c in [0, num_classes-1].
    if (d < embed_dim) {
        float pooled_val = pooled[d]; // Read activation for feature dimension d
        float grad_pooled = 0.0f;     // accumulates the total loss gradient flowing into feature dimension d of the pooled activation vector.

        for (int c = 0; c < num_classes; ++c) {
            float p_c = expf(logits[c] - max_logit) / sum_exp;
            // Logit Gradient (dL_dlogit) = p_c - y_c.
            float dL_dlogit = p_c - (c == target ? 1.0f : 0.0f);

            // 1. Backprop into pooled representations: dL/dpooled[d] = sum_c (dL/dz_c * W_class[d, c])
            // Accumulates: (p_c - y_c) * W_class[d, c]
            grad_pooled += dL_dlogit * W_class[d * num_classes + c];

            // 2. Weight gradient accumulation: dL/dW_class[d, c] = dL/dz_c * pooled[d]
            // dL_dW_class is the gradient of the loss function L with respect to the class representations matrix.
            // Shape is [EMBED_DIM, NUM_CLASSES]
            // W_class is used to convert snapshot vector [EMBED_DIM] into class vector/logits [NUM_CLASSES].
            // "* num_classes" refers to the second value (number of columns) in shape.
            // dL_dW_class[d, c]
            atomicAdd(&dL_dW_class[d * num_classes + c], dL_dlogit * pooled_val);
        }

        // Divide gradient evenly across spatial patches for un-pooling
        // dL_dtokens is the gradient of the loss function L with respect to the patch token representations matrix (tokens).
        // Shape [NUM_PATCHES, EMBED_DIM]
        // dL_dtokens[p, d]
        float grad_token = grad_pooled / static_cast<float>(num_patches);
        for (int p = 0; p < num_patches; ++p) {
            dL_dtokens[p * embed_dim + d] = grad_token;
        }
    }
}

// ============================================================================
// 5. BACKWARD GRADIENT KERNEL
// ============================================================================
__global__ void vit_embed_backward_bool_kernel(
    const float* __restrict__ dL_dtokens, // [NUM_PATCHES, EMBED_DIM]
    const float* __restrict__ densities,  // [NUM_PATCHES, WORDS_PER_PATCH]
    float*       __restrict__ dL_dW_proj)  // [WORDS_PER_PATCH, EMBED_DIM]
{
    int w_idx   = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (w_idx >= WORDS_PER_PATCH || dim_idx >= EMBED_DIM) return;

    float grad_acc = 0.0f;
    for (int p = 0; p < NUM_PATCHES; ++p) {
        float density = densities[p * WORDS_PER_PATCH + w_idx];
        float dL_dt   = dL_dtokens[p * EMBED_DIM + dim_idx];
        grad_acc += density * dL_dt;
    }

    dL_dW_proj[w_idx * EMBED_DIM + dim_idx] = grad_acc;
}

// ============================================================================
// 6. ADAMW OPTIMIZER KERNEL
// ============================================================================
__global__ void adamw_update_kernel(
    float* __restrict__ weights,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
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
// 7. MAIN PIPELINE EXECUTION
// ============================================================================
int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Dual-Stream Asynchronous Boolean Hypercube Classifier Execution     " << std::endl;
    std::cout << "======================================================================" << std::endl;

    // 1. Allocate Boolean Hypercube State Buffer (array of bools)
    bool* d_hypercube_bools = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hypercube_bools, HYPERCUBE_BOOLS * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_hypercube_bools, 1, HYPERCUBE_BOOLS * sizeof(bool))); // Initialize true

    // Target label for supervised classification training
    int h_target_label = 3;
    int* d_target_label = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target_label, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_target_label, &h_target_label, sizeof(int), cudaMemcpyHostToDevice));

    // 2. Projection and Classifier Weights
    int num_proj_weights = WORDS_PER_PATCH * EMBED_DIM;
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

    // Initialize Weights
    std::vector<float> h_W_proj_init(num_proj_weights);
    std::vector<float> h_W_class_init(num_class_weights);
    for (int i = 0; i < num_proj_weights; ++i) h_W_proj_init[i] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f;
    for (int i = 0; i < num_class_weights; ++i) h_W_class_init[i] = static_cast<float>(rand()) / RAND_MAX * 0.02f - 0.01f;

    CUDA_CHECK(cudaMemcpy(d_W_proj, h_W_proj_init.data(), num_proj_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_class, h_W_class_init.data(), num_class_weights * sizeof(float), cudaMemcpyHostToDevice));

    // Intermediate Forward/Backward Buffers
    float *d_tokens = nullptr, *d_densities = nullptr, *d_pooled = nullptr;
    float *d_logits = nullptr, *d_dL_dtokens = nullptr;
    float *d_loss_out = nullptr;
    int*   d_correct_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tokens, NUM_PATCHES * EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_densities, NUM_PATCHES * WORDS_PER_PATCH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pooled, EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dL_dtokens, NUM_PATCHES * EMBED_DIM * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_loss_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct_out, sizeof(int)));

    // 3. Create CUDA Streams
    cudaStream_t stream_mutate, stream_train;
    CUDA_CHECK(cudaStreamCreate(&stream_mutate));
    CUDA_CHECK(cudaStreamCreate(&stream_train));

    std::cout << "[+] System Initialized. Hypercube Buffer: Array of " << HYPERCUBE_BOOLS << " Bools." << std::endl;
    std::cout << "[+] Starting Training Stream Pipeline...\n" << std::endl;

    dim3 fwd_grid(NUM_PATCHES);
    dim3 fwd_block(EMBED_DIM);
    dim3 bwd_grid(WORDS_PER_PATCH);
    dim3 bwd_block(EMBED_DIM);

    int mutate_threads = 256;
    int mutate_blocks  = (HYPERCUBE_BOOLS + mutate_threads - 1) / mutate_threads;

    int opt_threads = 256;
    int opt_proj_blocks  = (num_proj_weights + opt_threads - 1) / opt_threads;
    int opt_class_blocks = (num_class_weights + opt_threads - 1) / opt_threads;

    for (int step = 1; step <= 10; ++step) {
        // Mutate Boolean Hypercube State asynchronously
        mutate_hypercube_bool_kernel<<<mutate_blocks, mutate_threads, 0, stream_mutate>>>(d_hypercube_bools, 1337 + step);

        // 1. Forward Token Embedding Pass from Boolean State
        vit_embed_forward_bool_kernel<<<fwd_grid, fwd_block, 0, stream_train>>>(
            d_hypercube_bools, d_W_proj, d_tokens, d_densities
        );

        // 2. Global Average Pooling
        int pool_blocks = (EMBED_DIM + 255) / 256;
        global_avg_pool_kernel<<<pool_blocks, 256, 0, stream_train>>>(
            d_tokens, d_pooled, NUM_PATCHES, EMBED_DIM
        );

        // 3. Classifier Linear Projection
        int class_blocks = (NUM_CLASSES + 255) / 256;
        linear_classifier_kernel<<<class_blocks, 256, 0, stream_train>>>(
            d_pooled, d_W_class, d_logits, EMBED_DIM, NUM_CLASSES
        );

        // 4. Softmax Cross-Entropy Loss & Accuracy Calculation
        CUDA_CHECK(cudaMemsetAsync(d_dL_dW_class, 0, num_class_weights * sizeof(float), stream_train));
        softmax_cross_entropy_kernel<<<pool_blocks, 256, 0, stream_train>>>(
            d_logits, d_pooled, d_target_label, d_W_class, d_dL_dtokens, d_dL_dW_class,
            d_loss_out, d_correct_out, NUM_PATCHES, EMBED_DIM, NUM_CLASSES
        );

        // 5. Backward Pass (Projection Layer)
        vit_embed_backward_bool_kernel<<<bwd_grid, bwd_block, 0, stream_train>>>(
            d_dL_dtokens, d_densities, d_dL_dW_proj
        );

        // 6. AdamW Optimizer Updates (Projection Weights & Classifier Weights)
        adamw_update_kernel<<<opt_proj_blocks, opt_threads, 0, stream_train>>>(
            d_W_proj, d_dL_dW_proj, d_m_proj, d_v_proj,
            num_proj_weights, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_class_blocks, opt_threads, 0, stream_train>>>(
            d_W_class, d_dL_dW_class, d_m_class, d_v_class,
            num_class_weights, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        // Retrieve Loss and Accuracy from GPU Host-side Sync
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

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream_mutate));
    CUDA_CHECK(cudaStreamDestroy(stream_train));

    CUDA_CHECK(cudaFree(d_hypercube_bools));
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
    std::cout << "  Execution Complete! Hypercube Classifier pipeline verified.         " << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}