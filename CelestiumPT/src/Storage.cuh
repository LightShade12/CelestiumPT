#pragma once

#include <vector_types.h>
#include <glad/include/glad/glad.h>//for cudagl_interop
#include <cuda_gl_interop.h>
#include <cstdint>

struct Triangle;
struct ShapeIntersection;
struct SceneGeometry;
class DeviceCamera;
class Light {};

struct FrameBufferStorage {
public:
	int2 resolution;
	cudaSurfaceObject_t composite_render_surface_object;
	cudaSurfaceObject_t normals_render_surface_object;
	cudaSurfaceObject_t positions_render_surface_object;
	cudaSurfaceObject_t albedo_render_surface_object;
	float3* accumulation_framebuffer = nullptr;
};

struct DeviceSceneDescriptor {
	template<typename T>
	struct DeviceBuffer {
		const T* data = nullptr;
		size_t size = 0;
	};

	SceneGeometry* dev_aggregate = nullptr;
	DeviceBuffer<Light>dev_lights;
	DeviceBuffer<Light>dev_inf_lights;
	DeviceCamera* dev_camera = nullptr;
};

struct IntegratorSettings {
	bool accumulate = true;
	int bounces = 2;
	bool MIS = false;
};

struct IntegratorGlobals {
	FrameBufferStorage FrameBuffer;
	DeviceSceneDescriptor SceneDescriptor;
	IntegratorSettings IntegratorCFG;
	uint32_t frameidx = 0;
};