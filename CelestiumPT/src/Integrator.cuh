#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <glad/include/glad/glad.h>
#include <cuda_gl_interop.h>

#include <cstdint>

class Primitive {};
class Light {};
class Camera {};

class FrameBufferStorage {
	cudaSurfaceObject_t composite_render_surface_object;
	cudaSurfaceObject_t normals_render_surface_object;
	cudaSurfaceObject_t albedo_render_surface_object;
};

template<typename T>
struct DeviceBuffer {
	const T* data = nullptr;
	size_t size = 0;
};

struct DeviceSceneDescriptor {
	Primitive* dev_aggregate = nullptr;

	DeviceBuffer<Light>dev_lights;
	DeviceBuffer<Light>dev_inf_lights;
	Camera* dev_camera = nullptr;
};

struct IntegratorSettings {
};

struct IntegratorGlobals {
	FrameBufferStorage FrameBuffer;
	DeviceSceneDescriptor SceneDescriptor;
	IntegratorSettings IntegratorCFG;
	uint32_t frameidx = 0;
};

__global__ void renderKernel(IntegratorGlobals globals, cudaSurfaceObject_t composite_render_surface_obj, int frame_width, int frame_height);

namespace IntegratorPipeline {
	void Initialize();

	void invokeRenderKernel(IntegratorGlobals globals, cudaSurfaceObject_t composite_render_surface_obj, dim3 block_grid_dims, dim3 thread_block_dims,
		int frame_width, int frame_height);

	void Render();

	__device__ float3 evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel, int frame_width, int frame_height);

	__device__ void Intersect();
	__device__ bool IntersectP();
	__device__ bool Unoccluded();
	__device__ float3 Li();
	__device__ float3 LiRandomWalk();
}