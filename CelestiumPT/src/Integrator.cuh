#pragma once

#include "storage.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

//struct IntegratorGlobals;
class Ray;
class RGBSpectrum;
class BSDF;
class LightSampler;

// Path tracing kernel that writes irradiance, moment data, and G-buffer
__global__ void tracePathSample(const IntegratorGlobals globals);
__device__ void computeVelocity(const IntegratorGlobals& globals, float2 tc_uv, int2 ppixel);
__device__ RGBSpectrum staticAccumulation(const IntegratorGlobals& globals, RGBSpectrum radiance_sample, int2 c_pix);

namespace IntegratorPipeline {
	__device__ RGBSpectrum evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel);
	__device__ RGBSpectrum deferredEvaluatePixelSample(const IntegratorGlobals& globals, int2 ppixel, uint32_t seed);

	//TraceRay function
	__device__ ShapeIntersection Intersect(const IntegratorGlobals& globals, const Ray& ray, float tmax = FLT_MAX);
	__device__ bool IntersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax);

	__device__ bool Unoccluded(const IntegratorGlobals& globals, const ShapeIntersection& p0, float3 p1);

	__device__ RGBSpectrum LiPathIntegrator(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel);

	__device__ RGBSpectrum SampleLd(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& payload,
		const BSDF& bsdf, const LightSampler& light_sampler, uint32_t& seed, bool primary_surface);
}