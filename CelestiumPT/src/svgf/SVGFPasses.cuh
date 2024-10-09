#pragma once
#include "storage.cuh"
#define __CUDACC__
#include <cuda_runtime.h>

// SVGF kernel that applies spatial filtering
__global__ void SVGFPass(const IntegratorGlobals globals, int stepsize);

__device__ float spatialVarianceEstimate(const IntegratorGlobals& globals, int2 t_current_pix);