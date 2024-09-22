#pragma once

#include <cuda_runtime.h>
#include "maths/vector_maths.cuh"
#include "maths/constants.cuh"

namespace Samplers {
	__device__ uint32_t pcg_hash(uint32_t input);
	//0-1
	__device__ float randF_PCGHash(uint32_t& seed);

	__device__ float get1D_PCGHash(uint32_t& seed);
	__device__ float2 get2D_PCGHash(uint32_t& seed);
	__device__ float2 getPixel2D_PCGHash(uint32_t& seed);

	__device__ float3 sampleUniformSphere(float2 seed);
	//TODO: replace with tangent space version
	__device__ float3 sampleCosineWeightedHemisphere(float2 xi);
}