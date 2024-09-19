#pragma once

#include "Storage.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

//struct IntegratorGlobals;
class Ray;
class RGBSpectrum;

__global__ void renderKernel(IntegratorGlobals globals);

namespace IntegratorPipeline {
	//wrapper for kernel launch
	__host__  void invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims);

	__device__ RGBSpectrum evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel);

	//TraceRay function
	__device__ ShapeIntersection Intersect(const IntegratorGlobals& globals, const Ray& ray);
	__device__ bool IntersectP(const IntegratorGlobals& globals, const Ray& ray);
	__device__ bool Unoccluded(const IntegratorGlobals& globals, const Ray& ray);

	__device__ RGBSpectrum Li(const IntegratorGlobals& globals, const Ray& ray, uint32_t seed, float2 ppixel);
	__device__ RGBSpectrum LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel);
}