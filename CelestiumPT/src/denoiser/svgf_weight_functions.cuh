#pragma once

#define __CUDACC__
#include <cuda_runtime.h>
#include <vector_types.h>

// Normal-weighting function (Section 4.4.1)
__device__ float normalWeight(float3 normal0, float3 normal1);

// Depth-weighting function (Section 4.4.2)
__device__ float depthWeight(float depth0, float depth1, float2 grad, float2 offset);

// Luminance-weighting function (Section 4.4.3)
__device__ float luminanceWeight(float lum0, float lum1, float variance);