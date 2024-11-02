#pragma once

#include "storage.cuh"

#define __CUDACC__
#include <cuda_runtime.h>

//Launch with thread dims 16x16=256
__global__ void computeHistogram(const IntegratorGlobals t_globals);

//launch with thread dims= 256 x 1;
__global__ void computeAverageLuminance(const IntegratorGlobals t_globals);

__global__ void toneMap(const IntegratorGlobals t_globals);