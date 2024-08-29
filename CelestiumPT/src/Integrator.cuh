#pragma once
#include "maths/vector_types_extension.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>
#include <glad/include/glad/glad.h>
#include <cuda_gl_interop.h>

#include <cstdint>

class Primitive {};
class Light {};

class Ray {
public:

	Ray(float3 orig, float3 dir) :origin(orig), direction(dir), invdirection(1.f / dir) {};

	void setOrigin(float3 new_orig) { origin = new_orig; };
	void setDirection(float3 new_dir) { direction = new_dir; invdirection = 1.f / direction; };

	inline float3 getOrigin() const { return origin; };
	inline float3 getDirection() const { return direction; };
	inline float3 getInvDirection() const { return invdirection; };
private:
	float3 origin;
	float3 direction;
	float3 invdirection;
};

class Camera {
public:
	float3 origin;
};

struct FrameBufferStorage {
public:
	float2 resolution;
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
	Camera* dev_camera = nullptr;
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
	void invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims);

	__device__ float3 evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel);

	__device__ void Intersect();
	__device__ bool IntersectP();
	__device__ bool Unoccluded();
	__device__ float3 Li();
	__device__ float3 LiRandomWalk();
}