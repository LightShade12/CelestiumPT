#pragma once
#include "light.cuh"

struct SampledLight {
	__device__ SampledLight() = default;
	__device__ SampledLight(const Light* light, float p) :light(light), p(p) {};
	const Light* light = nullptr;
	float p = 0;

	__device__ operator bool() const {
		return (light != nullptr);
	}
	__device__ bool operator !() {
		return (light == nullptr);
	}
};

//Uniform light sampler
class LightSampler {
public:
	__device__ LightSampler(const Light* lights_buffer, size_t lights_buffer_size) :
		LightsBuffer(lights_buffer), LightsBufferSize(lights_buffer_size) {};

	__device__ SampledLight sample(float u) const {
		if (LightsBufferSize < 1) return {};
		int light_index = min(int(u * LightsBufferSize), int(LightsBufferSize - 1));
		return SampledLight(&(LightsBuffer[light_index]), (1.f / LightsBufferSize));
	};

	__device__ float PMF(const Light* light) const {
		if (LightsBufferSize < 1)return 0;
		return 1.f / LightsBufferSize;
	}

	size_t LightsBufferSize = 0;
	const Light* LightsBuffer = nullptr;
};