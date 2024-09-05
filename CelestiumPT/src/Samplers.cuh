#pragma once
#include <cuda_runtime.h>

namespace Samplers {
	__device__ uint32_t pcg_hash(uint32_t input)
	{
		uint32_t state = input * 747796405u + 2891336453u;
		uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
		return (word >> 22u) ^ word;
	}
	//0-1
	__device__ float randF_PCGHash(uint32_t& seed)
	{
		seed = pcg_hash(seed);
		return (float)seed / (float)UINT32_MAX;
	}

	__device__ float get1D_PCGHash(uint32_t& seed) { return randF_PCGHash(seed); };
	__device__ float2 get2D_PCGHash(uint32_t& seed) { return make_float2(get1D_PCGHash(seed), get1D_PCGHash(seed)); };
	__device__ float2 getPixel2D_PCGHash(uint32_t& seed) { return get2D_PCGHash(seed); };

	//TODO: replace with tangent space version
	__device__ float3 sampleCosineWeightedHemisphere(const float3& normal, float2 xi) {
		// Generate a cosine-weighted direction in the local frame
		float phi = 2.0f * PI * xi.x;
		float cosTheta = sqrtf(xi.y);//TODO: might have to switch with sinTheta
		float sinTheta = sqrtf(1.0f - xi.y);

		float3 H;
		H.x = sinTheta * cosf(phi);
		H.y = sinTheta * sinf(phi);
		H.z = cosTheta;

		// Create an orthonormal basis (tangent, bitangent, normal)
		float3 up = fabs(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
		float3 tangent = normalize(cross(up, normal));
		float3 bitangent = cross(normal, tangent);

		// Transform the sample direction from local space to world space
		return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
	}
}