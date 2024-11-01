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
	cudaSurfaceObject_t bloom_surfobject;
	cudaSurfaceObject_t mip0_surfobject;
	cudaSurfaceObject_t mip1_surfobject;
	cudaSurfaceObject_t mip2_surfobject;
	cudaSurfaceObject_t mip3_surfobject;
	cudaSurfaceObject_t mip4_surfobject;
	cudaSurfaceObject_t mip5_surfobject;

	//raw
	cudaSurfaceObject_t raw_irradiance_surfobject;

	//temporal accum
	cudaSurfaceObject_t integrated_irradiance_front_surfobject;//read only
	cudaSurfaceObject_t integrated_irradiance_back_surfobject;//write only
	cudaSurfaceObject_t integrated_moments_front_surfobject;//read only
	cudaSurfaceObject_t integrated_moments_back_surfobject;//write only

	//asvgf
	cudaSurfaceObject_t asvgf_gradient_sample_surfobject;
	cudaSurfaceObject_t asvgf_sparse_gradient_surfobject;
	cudaSurfaceObject_t asvgf_dense_gradient_front_surfobject;
	cudaSurfaceObject_t asvgf_dense_gradient_back_surfobject;

	cudaSurfaceObject_t asvgf_luminance_front_surfobject;
	cudaSurfaceObject_t asvgf_luminance_back_surfobject;
	cudaSurfaceObject_t asvgf_variance_back_surfobject;
	cudaSurfaceObject_t asvgf_variance_front_surfobject;

	cudaSurfaceObject_t history_shading_surfobject;
	cudaSurfaceObject_t history_seeds_surfobject;
	cudaSurfaceObject_t seeds_surfobject;
	cudaSurfaceObject_t history_triangleID_surfobject;
	cudaSurfaceObject_t history_objectID_surfobject;

	cudaSurfaceObject_t history_local_normals_surfobject;
	cudaSurfaceObject_t history_world_positions_surfobject;
	cudaSurfaceObject_t history_local_positions_surfobject;
	cudaSurfaceObject_t history_UVs_surfobject;
	cudaSurfaceObject_t history_bary_surfobject;

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
	cudaSurfaceObject_t triangleID_surfobject;
	cudaSurfaceObject_t velocity_surfobject;

	//debugviews
	cudaSurfaceObject_t debugview_GAS_overlap_surfobject;
	cudaSurfaceObject_t debugview_tri_test_heatmap_surfobject;
	cudaSurfaceObject_t debugview_bbox_test_heatmap_surfobject;
	cudaSurfaceObject_t debugview_objectID_surfobject;
	cudaSurfaceObject_t debugview_misc_surfobject;

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
	//must be 256 in size
	uint* GlobalHistogramBuffer = nullptr;//cuz we need persistent data bewteen kernels
	float* AverageLuminance = nullptr;//same plus atomics
};