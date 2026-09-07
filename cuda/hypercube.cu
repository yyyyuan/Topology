
__global__ 
void run(int N, bool *vertex_internal_state, bool *vertex_excited_state,
                    int *energy, int *vertex_upper_bound, int *vertex_lower_bound,
                    int *vertex_neighbor_index,
                    volatile int *__restrict__ control_flag // Used to safely stop the loop from CPU
)
{
    while (true)
    {
        energy[0]++;
    }

    return;
}

int main(void)
{
    int N = 1 << 22; // 4M elements

    bool *vertex_internal_state;
    bool *vertex_excited_state;
    int *energy;
    int *vertex_upper_bound;
    int *vertex_lower_bound;
    int *vertex_neighbor_index;

    int *control_flag = 0;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&vertex_internal_state, N * sizeof(bool));
    cudaMallocManaged(&vertex_excited_state, N * sizeof(bool));
    cudaMallocManaged(&energy, N * sizeof(int));
    cudaMallocManaged(&vertex_upper_bound, N * sizeof(int));
    cudaMallocManaged(&vertex_lower_bound, N * sizeof(int));
    cudaMallocManaged(&vertex_neighbor_index, N * sizeof(int));

    int block_count = 100;
    int block_size = 256;

    run<<<1, 1>>>(N, vertex_internal_state, vertex_excited_state, energy,
                  vertex_upper_bound, vertex_lower_bound, vertex_neighbor_index, control_flag);

    // Free memory
    cudaFree(vertex_internal_state);
    cudaFree(vertex_excited_state);
    cudaFree(energy);
    cudaFree(vertex_upper_bound);
    cudaFree(vertex_lower_bound);
    cudaFree(vertex_neighbor_index);

    return 0;
}