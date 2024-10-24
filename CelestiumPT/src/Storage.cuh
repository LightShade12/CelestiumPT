#pragma once

#include "integrator_settings.hpp"
#include "light.cuh"

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
	//post
	cudaSurfaceObject_t composite_surfobject;

	//raw
	cudaSurfaceObject_t raw_irradiance_surfobject;
	cudaSurfaceObject_t raw_moments_surfobject;
	
	//temporal accum
	cudaSurfaceObject_t integrated_irradiance_front_surfobject;//read only
	cudaSurfaceObject_t integrated_irradiance_back_surfobject;//write only
	cudaSurfaceObject_t integrated_moments_front_surfobject;//read only
	cudaSurfaceObject_t integrated_moments_back_surfobject;//write only

	//svgf
	cudaSurfaceObject_t svgf_filtered_variance_back_surfobject;
	cudaSurfaceObject_t svgf_filtered_variance_front_surfobject;
	cudaSurfaceObject_t svgf_filtered_irradiance_front_surfobject;//not to be fedback; svgf consumption only
	cudaSurfaceObject_t svgf_filtered_irradiance_back_surfobject;

	//gbuffer
	cudaSurfaceObject_t albedo_surfobject;
	cudaSurfaceObject_t world_normals_surfobject;
	cudaSurfaceObject_t local_normals_surfobject;
	cudaSurfaceObject_t world_positions_surfobject;
	cudaSurfaceObject_t local_positions_surfobject;
	cudaSurfaceObject_t depth_surfobject;
	cudaSurfaceObject_t UVs_surfobject;
	cudaSurfaceObject_t bary_surfobject;
	cudaSurfaceObject_t objectID_surfobject;
	cudaSurfaceObject_t velocity_surfobject;

	//debugviews
	cudaSurfaceObject_t debugview_GAS_overlap_surfobject;
	cudaSurfaceObject_t debugview_tri_test_heatmap_surfobject;
	cudaSurfaceObject_t debugview_bbox_test_heatmap_surfobject;
	cudaSurfaceObject_t debugview_objectID_surfobject;

	//history
	cudaSurfaceObject_t history_depth_surfobject;
	cudaSurfaceObject_t history_world_normals_surfobject;

	//static accum
	float3* accumulation_framebuffer = nullptr;
	float3* variance_accumulation_framebuffer = nullptr;
};

struct DeviceSceneDescriptor {
	SceneGeometry* DeviceGeometryAggregate = nullptr;
	DeviceCamera* ActiveCamera = nullptr;
};

struct IntegratorGlobals {
	FrameBufferStorage FrameBuffer;
	DeviceSceneDescriptor SceneDescriptor;
	IntegratorSettings IntegratorCFG;
	uint32_t FrameIndex = 0;
};