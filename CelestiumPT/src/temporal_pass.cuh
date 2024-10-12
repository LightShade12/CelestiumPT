#pragma once
#include "storage.cuh"

#define __CUDACC__
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>

//used for pure TAA samples
__global__ void temporalAccumulate(const IntegratorGlobals globals);

__device__ bool rejectionHeuristic(const IntegratorGlobals& globals, int2 prev_pix, int2 cur_px);