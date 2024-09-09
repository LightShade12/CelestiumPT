#pragma once

#include "maths/vector_maths.cuh"
#include "maths/constants.cuh"

struct BSDFSample {
	float3 f{};
	float3 wi{};
	float pdf;
};

class BSDF {
public:
	__device__ float3 f(float3 wo, float3 wi) {
		return  fOpaqueDielectric();
	}

	__device__ float pdf() {
		return 0.f;
	}

	__device__ BSDFSample sampleBSDF(float3 wo) {
		BSDFSample();
	}

	__device__ float3 fOpaqueDielectric() {
		return make_float3(0.8f) / PI;
	}
};