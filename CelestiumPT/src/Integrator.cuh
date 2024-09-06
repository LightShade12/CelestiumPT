#pragma once
#include "maths/maths_linear_algebra.cuh"
#include "SceneGeometry.cuh"
#include "DeviceCamera.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

struct IntegratorGlobals;

__global__ void renderKernel(IntegratorGlobals globals);

namespace IntegratorPipeline {
	//wrapper for kernel launch
	void invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims);

	__device__ float3 evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel);

	//TraceRay function
	__device__ ShapeIntersection Intersect(const IntegratorGlobals& globals, const Ray& ray);
	__device__ bool IntersectP(const IntegratorGlobals& globals, const Ray& ray);
	__device__ bool Unoccluded(const IntegratorGlobals& globals, const Ray& ray);

	__device__ float3 Li(const IntegratorGlobals& globals, const Ray& ray, uint32_t seed, float2 ppixel);
	__device__ float3 LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel);
}