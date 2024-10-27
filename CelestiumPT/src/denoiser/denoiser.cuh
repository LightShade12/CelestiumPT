#pragma once
#include "storage.cuh"
#define __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//cross-utility
__device__ bool rejectionHeuristic(const IntegratorGlobals& t_globals, int2 t_prev_pix, int2 t_current_pix);

// SVGF kernel that applies spatial filtering
__global__ void atrousSVGF(const IntegratorGlobals t_globals, int t_stepsize);

// Temporal integration kernel for SVGF; out: filtered_irradiance frontbuff; feedsback moments
__global__ void temporalAccumulate(const IntegratorGlobals t_globals);

__global__ void estimateVariance(const IntegratorGlobals t_globals);

__global__ void mergeSamples(const IntegratorGlobals t_globals);

__global__ void createGradientSamples(const IntegratorGlobals t_globals);

__global__ void atrousGradient(const IntegratorGlobals t_globals, int t_stepsize);