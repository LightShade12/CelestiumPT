#pragma once
#include "storage.cuh"
#define __CUDACC__
#include <cuda_runtime.h>

// SVGF kernel that applies spatial filtering
__global__ void atrousSVGF(const IntegratorGlobals globals, int stepsize);

// Temporal integration kernel for SVGF; out: filtered_irradiance frontbuff; feedsback moments
__global__ void temporalAccumulate(const IntegratorGlobals globals);