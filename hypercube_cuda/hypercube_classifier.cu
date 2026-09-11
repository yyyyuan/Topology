%%writefile main.cu

#include "hypercube_classifier.cuh"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

// ============================================================================
// 1. RAW SIMULATION BUFFER MUTATION KERNEL
// ============================================================================
__global__ void mutate_raw_hypercube_kernel(
    bool* __restrict__ raw_hypercube, // Shape: [RAW_FRAMES, HYPERCUBE_BOOLS]
    uint32_t seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bools = RAW_FRAMES * HYPERCUBE_BOOLS;
    if (tid >= HYPERCUBE_BOOLS) return;

    bool current_state = raw_hypercube[tid];

    for (int t = 0; t < RAW_FRAMES; ++t) {
        if (t > 0) {
            uint32_t x = tid ^ (seed + t * 0x9e3779b9);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            current_state = (x & 0x1) ? !current_state : current_state;
        }
        raw_hypercube[t * HYPERCUBE_BOOLS + tid] = current_state;
    }
}

// ============================================================================
// 2. SPATIOTEMPORAL PROJECTION KERNEL
// ============================================================================
__global__ void spatiotemporal_projection_kernel(
    const bool* __restrict__ raw_hypercube, // [RAW_FRAMES, HYPERCUBE_BOOLS]
    const float* __restrict__ W_proj,        // [WORDS_PER_PATCH, EMBED_DIM]
    float*       __restrict__ patch_tokens,  // [TEMPORAL_STEPS, NUM_PATCHES, EMBED_DIM]
    float*       __restrict__ densities)     // [TEMPORAL_STEPS, NUM_PATCHES, WORDS_PER_PATCH]
{
    int step_idx  = blockIdx.y;  // [0, TEMPORAL_STEPS - 1]
    int patch_idx = blockIdx.x;  // [0, NUM_PATCHES - 1]
    int dim_idx   = threadIdx.x; // [0, EMBED_DIM - 1]

    if (step_idx >= TEMPORAL_STEPS || patch_idx >= NUM_PATCHES || dim_idx >= EMBED_DIM) return;

    int token_offset = (step_idx * NUM_PATCHES + patch_idx);
    float total_projection = 0.0f;

    for (int tube_f = 0; tube_f < TUBELET_FRAMES; ++tube_f) {
        int raw_frame_idx = step_idx * TUBELET_FRAMES + tube_f;

        const bool* frame_patch_bools = raw_hypercube +
            (raw_frame_idx * HYPERCUBE_BOOLS) + (patch_idx * BOOLS_PER_FRAME_PATCH);

        const uchar4* vec_ptr = reinterpret_cast<const uchar4*>(frame_patch_bools);
        int num_vecs = BOOLS_PER_FRAME_PATCH / 4;

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
__global__ void spatial_avg_pool_kernel(
    const float* __restrict__ patch_tokens,    // [TEMPORAL_STEPS, NUM_PATCHES, EMBED_DIM]
    float*       __restrict__ temporal_tokens) // [TEMPORAL_STEPS, EMBED_DIM]
{
    int step_idx = blockIdx.x;
    int dim_idx  = threadIdx.x;

    if (step_idx >= TEMPORAL_STEPS || dim_idx >= EMBED_DIM) return;

    float sum = 0.0f;
    for (int p = 0; p < NUM_PATCHES; ++p) {
        int token_offset = step_idx * NUM_PATCHES + p;
        sum += patch_tokens[token_offset * EMBED_DIM + dim_idx];
    }

    temporal_tokens[step_idx * EMBED_DIM + dim_idx] = sum / static_cast<float>(NUM_PATCHES);
}

// ============================================================================
// 4. STREAMING TEMPORAL SELF-ATTENTION KERNEL
// ============================================================================
__global__ void temporal_self_attention_kernel(
    const float* __restrict__ temporal_in,   // [TEMPORAL_STEPS, EMBED_DIM]
    const float* __restrict__ W_q,           // [EMBED_DIM, EMBED_DIM]
    const float* __restrict__ W_k,           // [EMBED_DIM, EMBED_DIM]
    const float* __restrict__ W_v,           // [EMBED_DIM, EMBED_DIM]
    float*       __restrict__ temporal_out, // [TEMPORAL_STEPS, EMBED_DIM]
    float*       __restrict__ attn_map_out) // [TEMPORAL_STEPS, TEMPORAL_STEPS]
{
    __shared__ float Q_step[EMBED_DIM];
    __shared__ float V[TEMPORAL_STEPS][EMBED_DIM];
    __shared__ float A_step[TEMPORAL_STEPS];

    int step_idx = blockIdx.x;
    int dim_idx  = threadIdx.x;

    if (step_idx >= TEMPORAL_STEPS || dim_idx >= EMBED_DIM) return;

    // Phase 1: Compute Q for step_idx and cache V matrix
    float q_val = 0.0f;
    for (int d = 0; d < EMBED_DIM; ++d) {
        q_val += temporal_in[step_idx * EMBED_DIM + d] * W_q[d * EMBED_DIM + dim_idx];
    }
    Q_step[dim_idx] = q_val;

    for (int t = 0; t < TEMPORAL_STEPS; ++t) {
        float v_val = 0.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            v_val += temporal_in[t * EMBED_DIM + d] * W_v[d * EMBED_DIM + dim_idx];
        }
        V[t][dim_idx] = v_val;
    }
    __syncthreads();

    // Phase 2: Compute Attention Scores A_step[t]
    if (dim_idx < TEMPORAL_STEPS) {
        int target_step = dim_idx;
        float score = 0.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            float k_val = 0.0f;
            for (int k_d = 0; k_d < EMBED_DIM; ++k_d) {
                k_val += temporal_in[target_step * EMBED_DIM + k_d] * W_k[k_d * EMBED_DIM + d];
            }
            score += Q_step[d] * k_val;
        }
        A_step[target_step] = score / sqrtf(static_cast<float>(EMBED_DIM));
    }
    __syncthreads();

    // Phase 3: Softmax over A_step
    if (dim_idx == 0) {
        float max_score = A_step[0];
        for (int j = 1; j < TEMPORAL_STEPS; ++j) {
            if (A_step[j] > max_score) max_score = A_step[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            A_step[j] = expf(A_step[j] - max_score);
            sum_exp += A_step[j];
        }
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            A_step[j] /= sum_exp;
            if (attn_map_out != nullptr) {
                attn_map_out[step_idx * TEMPORAL_STEPS + j] = A_step[j];
            }
        }
    }
    __syncthreads();

    // Phase 4: Output Context Vector
    float context = 0.0f;
    for (int j = 0; j < TEMPORAL_STEPS; ++j) {
        context += A_step[j] * V[j][dim_idx];
    }
    temporal_out[step_idx * EMBED_DIM + dim_idx] = context;
}

// ============================================================================
// 5. TEMPORAL AGGREGATION KERNEL
// ============================================================================
__global__ void temporal_avg_pool_kernel(
    const float* __restrict__ temporal_tokens, // [TEMPORAL_STEPS, EMBED_DIM]
    float*       __restrict__ pooled_seq)      // [EMBED_DIM]
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
__global__ void linear_classifier_kernel(
    const float* __restrict__ pooled_seq, // [EMBED_DIM]
    const float* __restrict__ W_class,    // [EMBED_DIM, NUM_CLASSES]
    float*       __restrict__ logits,     // [NUM_CLASSES]
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
    float*       __restrict__ dL_dW_v)        // [EMBED_DIM, EMBED_DIM]
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0, EMBED_DIM - 1]
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0, EMBED_DIM - 1]

    if (row >= EMBED_DIM || col >= EMBED_DIM) return;

    float grad_q_acc = 0.0f;
    float grad_k_acc = 0.0f;
    float grad_v_acc = 0.0f;

    float scale = 1.0f / sqrtf(static_cast<float>(EMBED_DIM));
    float dL_dattn_out = 1.0f / static_cast<float>(TEMPORAL_STEPS); // From average pooling over temporal steps

    for (int i = 0; i < TEMPORAL_STEPS; ++i) { // Query sequence step i
        float dL_dz_i_col = dL_dpooled[col] * dL_dattn_out;

        // 1. Gradient w.r.t W_v
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            float A_ij = attn_map[i * TEMPORAL_STEPS + j];
            grad_v_acc += A_ij * dL_dz_i_col * temporal_in[j * EMBED_DIM + row];
        }

        // 2. Gradient w.r.t Softmax Attention Scores (dL_dA_ij)
        // dL_dA_ij = sum_d (dL_dz_i_d * V_jd)
        for (int j = 0; j < TEMPORAL_STEPS; ++j) {
            float dL_dA_ij = 0.0f;
            for (int d = 0; d < EMBED_DIM; ++d) {
                float v_jd = 0.0f;
                for (int m = 0; m < EMBED_DIM; ++m) {
                    v_jd += temporal_in[j * EMBED_DIM + m] * W_v[m * EMBED_DIM + d];
                }
                dL_dA_ij += (dL_dpooled[d] * dL_dattn_out) * v_jd;
            }

            // Backprop through Softmax: dL_dS_ik = sum_j dL_dA_ij * A_ij * (delta_jk - A_ik)
            float A_ij = attn_map[i * TEMPORAL_STEPS + j];
            for (int k = 0; k < TEMPORAL_STEPS; ++k) {
                float A_ik = attn_map[i * TEMPORAL_STEPS + k];
                float delta_jk = (j == k) ? 1.0f : 0.0f;
                float dL_dS_ik = dL_dA_ij * A_ij * (delta_jk - A_ik) * scale;

                // k_kd = sum_m x_km * W_k_md
                float k_kd_col = 0.0f;
                for (int m = 0; m < EMBED_DIM; ++m) {
                    k_kd_col += temporal_in[k * EMBED_DIM + m] * W_k[m * EMBED_DIM + col];
                }
                grad_q_acc += dL_dS_ik * temporal_in[i * EMBED_DIM + row] * k_kd_col;

                // q_id = sum_m x_im * W_q_md
                float q_id_col = 0.0f;
                for (int m = 0; m < EMBED_DIM; ++m) {
                    q_id_col += temporal_in[i * EMBED_DIM + m] * W_q[m * EMBED_DIM + col];
                }
                grad_k_acc += dL_dS_ik * q_id_col * temporal_in[k * EMBED_DIM + row];
            }
        }
    }

    dL_dW_q[row * EMBED_DIM + col] = grad_q_acc;
    dL_dW_k[row * EMBED_DIM + col] = grad_k_acc;
    dL_dW_v[row * EMBED_DIM + col] = grad_v_acc;
}

__global__ void spatiotemporal_backward_kernel(
    const float* __restrict__ dL_dpooled,  // [EMBED_DIM]
    const float* __restrict__ densities,   // [TEMPORAL_STEPS, NUM_PATCHES, WORDS_PER_PATCH]
    float*       __restrict__ dL_dW_proj)   // [WORDS_PER_PATCH, EMBED_DIM]
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

    for (int step = 1; step <= 10; ++step) {
        mutate_raw_hypercube_kernel<<<mutate_blocks, 256>>>(d_raw_hypercube, 1337 + step);

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

        // Loss & Classification Backward Pass
        CUDA_CHECK(cudaMemset(d_dL_dW_class, 0, num_class_weights * sizeof(float)));
        softmax_cross_entropy_kernel<<<(EMBED_DIM + 255) / 256, 256>>>(
            d_logits, d_pooled_seq, d_target_label, d_W_class,
            d_dL_dpooled, d_dL_dW_class, d_loss_out, d_correct_out,
            EMBED_DIM, NUM_CLASSES
        );

        // Attention Backward Pass (dL/dW_q, dL/dW_k, dL/dW_v)
        temporal_attention_backward_kernel<<<attn_back_grid, attn_back_block>>>(
            d_dL_dpooled, d_temporal_tokens, d_attn_map,
            d_W_q, d_W_k, d_W_v,
            d_dL_dW_q, d_dL_dW_k, d_dL_dW_v
        );

        // Projection Layer Backward Pass
        spatiotemporal_backward_kernel<<<WORDS_PER_PATCH, EMBED_DIM>>>(
            d_dL_dpooled, d_densities, d_dL_dW_proj
        );

        // AdamW Parameter Updates
        adamw_update_kernel<<<opt_proj_blocks, 256>>>(
            d_W_proj, d_dL_dW_proj, d_m_proj, d_v_proj,
            num_proj_weights, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_class_blocks, 256>>>(
            d_W_class, d_dL_dW_class, d_m_class, d_v_class,
            num_class_weights, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_attn_blocks, 256>>>(
            d_W_q, d_dL_dW_q, d_m_q, d_v_q,
            num_attn_weights, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_attn_blocks, 256>>>(
            d_W_k, d_dL_dW_k, d_m_k, d_v_k,
            num_attn_weights, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, step
        );

        adamw_update_kernel<<<opt_attn_blocks, 256>>>(
            d_W_v, d_dL_dW_v, d_m_v, d_v_v,
            num_attn_weights, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, step
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

    // Cleanup
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

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  Spatio-Temporal Pipeline Execution Complete!                        " << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}