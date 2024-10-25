#pragma once

#include "storage.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

//struct IntegratorGlobals;
class Ray;
class RGBSpectrum;
class BSDF;
class LightSampler;

// Path tracing kernel that writes out irradiance
__global__ void tracePathSample(const IntegratorGlobals globals);

namespace IntegratorPipeline {
	__device__ RGBSpectrum evaluatePixelSample(const IntegratorGlobals& globals, int2 ppixel);

	__device__ RGBSpectrum deferredEvaluatePixelSample(const IntegratorGlobals& t_globals, int2 t_current_pix, uint32_t t_seed);

	//TraceRay function
	__device__ ShapeIntersection Intersect(const IntegratorGlobals& globals, const Ray& ray, float tmax = FLT_MAX);
	__device__ bool IntersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax);

	__device__ bool Unoccluded(const IntegratorGlobals& globals, const ShapeIntersection& p0, float3 p1);

	__device__ RGBSpectrum LiPathIntegrator(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, int2 ppixel);

	__device__ RGBSpectrum SampleLd(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& payload,
		const BSDF& bsdf, const LightSampler& light_sampler, uint32_t& seed, bool primary_surface);
}