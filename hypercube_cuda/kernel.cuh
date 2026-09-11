%%writefile kernel.cuh

#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

#include "hypercube_kernel_params.cuh"

__global__
void run(HypercubeKernelParams params);

#endif
