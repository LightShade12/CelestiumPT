#include "samplers.cuh"

namespace Samplers {
	__device__ uint32_t pcg_hash(uint32_t input)
	{
		uint32_t state = input * 747796405u + 2891336453u;
		uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		return (word >> 22u) ^ word;
	}
	__device__ float randF_PCGHash(uint32_t& seed)
	{
		seed = pcg_hash(seed);
		return (float)seed / (float)UINT32_MAX;
	}

	__device__ float get1D_PCGHash(uint32_t& seed)
	{
		return randF_PCGHash(seed);
	}

	__device__ float2 get2D_PCGHash(uint32_t& seed)
	{
		return make_float2(get1D_PCGHash(seed), get1D_PCGHash(seed));
	}

	__device__ float2 getPixel2D_PCGHash(uint32_t& seed)
	{
		return get2D_PCGHash(seed);
	}

	__device__ float3 sampleUniformSphere(float2 seed)
	{
		float z = 1 - 2 * seed.x;
		float r = fmaxf(0, sqrtf(1 - (z * z)));
		float phi = 2 * PI * seed.y;
		return { r * cosf(phi), r * sinf(phi), z };
	}

	__device__ float3 sampleCosineWeightedHemisphere(float2 xi)
	{
		// Generate a cosine-weighted direction in the local frame
		float phi = 2.0f * PI * xi.x;
		float cosTheta = sqrtf(xi.y);
		float sinTheta = sqrtf(1.0f - xi.y);

		float3 H;
		H.x = sinTheta * cosf(phi);
		H.y = sinTheta * sinf(phi);
		H.z = cosTheta;

		return H;
	}
}