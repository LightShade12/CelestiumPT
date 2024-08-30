#pragma once
#include "maths/maths_linear_algebra.cuh"
#include "Camera.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <glad/include/glad/glad.h>
#include <cuda_gl_interop.h>

#include <cstdint>

struct ShapeIntersection {
};

class Mesh {
	__device__ ShapeIntersection intersect(const Ray& ray);
	__device__ bool intersectP(const Ray& ray);
	Mat4 modelMatrix;
};

class Primitive {
	ShapeIntersection intersect(const Ray& ray);
	bool intersectP(const Ray& ray);
};
class Light {};

struct FrameBufferStorage {
public:
	int2 resolution;
	cudaSurfaceObject_t composite_render_surface_object;
	cudaSurfaceObject_t normals_render_surface_object;
	cudaSurfaceObject_t albedo_render_surface_object;
};

struct DeviceSceneDescriptor {
	template<typename T>
	struct DeviceBuffer {
		const T* data = nullptr;
		size_t size = 0;
	};

	Primitive* dev_aggregate = nullptr;
	DeviceBuffer<Light>dev_lights;
	DeviceBuffer<Light>dev_inf_lights;
	DeviceCamera* dev_camera = nullptr;
};

struct IntegratorSettings {
	bool accumulate = false;
};

struct IntegratorGlobals {
	FrameBufferStorage FrameBuffer;
	DeviceSceneDescriptor SceneDescriptor;
	IntegratorSettings IntegratorCFG;
	uint32_t frameidx = 0;
};

__global__ void renderKernel(IntegratorGlobals globals);

namespace IntegratorPipeline {
	//wrapper for kernel launch
	void invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims);

	__device__ float3 evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel);

	__device__ void Intersect();
	__device__ bool IntersectP();
	__device__ bool Unoccluded();
	__device__ float3 Li();
	__device__ float3 LiRandomWalk();
}