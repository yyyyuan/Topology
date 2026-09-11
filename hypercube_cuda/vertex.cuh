%%writefile vertex.cuh

#ifndef _VERTEX_H_
#define _VERTEX_H_

// The basic vertex used in the hypercube topology.
// Mimic the QM Particle I designed to unify the Theory of General Relativity and the Quantum Mechanics.
//
// It's called "Vertex" instead of of "Particle" to tell the naunce between this new terminology and the macro scale term "Particle".
// With this new term it tells explicitly that this is something different from the traditional or even QM particles in physics.
//
// The first step to build the new system.

#include <cstdint> // Required for int32_t
#include <vector>

#include "hypercube_kernel_params.cuh"

// Let there be Light.
__device__ void run_vertex(
  int index,
  const HypercubeKernelParams& params);

__device__ void spin(
  int index,
  int *vertex_neighbor_index);

// The equivical run_vertex function expressed in math format using bitwise operators only.
__device__ void run_vertex_math(int index, const HypercubeKernelParams& params);

#endif