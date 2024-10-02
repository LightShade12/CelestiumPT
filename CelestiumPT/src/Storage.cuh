#pragma once

#include "IntegratorSettings.hpp"
#include "Light.cuh"

#include <vector_types.h>
#include <glad/include/glad/glad.h>//for cudagl_interop
#include <cuda_gl_interop.h>
#include <cstdint>

struct Triangle;
struct ShapeIntersection;
struct SceneGeometry;
class DeviceCamera;

struct FrameBufferStorage {
public:
	int2 resolution;
	cudaSurfaceObject_t composite_render_surface_object;
	cudaSurfaceObject_t albedo_render_surface_object;
	cudaSurfaceObject_t world_normals_render_surface_object;
	cudaSurfaceObject_t local_normals_render_surface_object;
	cudaSurfaceObject_t positions_render_surface_object;
	cudaSurfaceObject_t local_positions_render_surface_object;
	cudaSurfaceObject_t GAS_debug_render_surface_object;
	cudaSurfaceObject_t depth_render_surface_object;
	cudaSurfaceObject_t UV_debug_render_surface_object;
	cudaSurfaceObject_t bary_debug_render_surface_object;
	cudaSurfaceObject_t objectID_render_surface_object;
	cudaSurfaceObject_t velocity_render_surface_object;
	cudaSurfaceObject_t objectID_debug_render_surface_object;
	cudaSurfaceObject_t history_integrated_irradiance_front_surfobj;//read only
	cudaSurfaceObject_t history_integrated_irradiance_back_surfobj;//write only
	cudaSurfaceObject_t history_depth_render_surface_object;
	cudaSurfaceObject_t history_world_normals_render_surface_object;
	float3* accumulation_framebuffer = nullptr;
};

struct DeviceSceneDescriptor {
	SceneGeometry* device_geometry_aggregate = nullptr;
	DeviceCamera* active_camera = nullptr;
};

struct IntegratorGlobals {
	FrameBufferStorage FrameBuffer;
	DeviceSceneDescriptor SceneDescriptor;
	IntegratorSettings IntegratorCFG;
	uint32_t frameidx = 0;
};