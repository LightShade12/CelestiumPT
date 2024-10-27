#pragma once
#include "storage.cuh"
#include "cuda_utility.cuh"
#include "film.cuh"
#include "device_camera.cuh"

#include "ray.cuh"
#include "scene_geometry.cuh"
#include "maths/linear_algebra.cuh"
#include "shape_intersection.cuh"
#include "device_mesh.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//velocity is in screen UV space
__device__ void computeVelocity(const IntegratorGlobals& t_globals, int2 t_current_pix)
{
	float c_objID = texReadNearest(t_globals.FrameBuffer.objectID_surfobject, t_current_pix).x;
	float4 c_lpos = texReadNearest(t_globals.FrameBuffer.local_positions_surfobject, t_current_pix);

	float3 l_pos = make_float3(c_lpos);

	if (c_objID < 0) {
		texWrite(make_float4(0, 0, 0, 1),
			t_globals.FrameBuffer.velocity_surfobject, t_current_pix);
		return;
	}

	Mat4 c_VP = t_globals.SceneDescriptor.ActiveCamera->projectionMatrix * t_globals.SceneDescriptor.ActiveCamera->viewMatrix;
	Mat4 p_VP = t_globals.SceneDescriptor.ActiveCamera->prev_projectionMatrix *
		t_globals.SceneDescriptor.ActiveCamera->prev_viewMatrix;
	Mat4 c_M = t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMeshesBuffer[(int)c_objID].modelMatrix;
	Mat4 p_M = t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMeshesBuffer[(int)c_objID].prev_modelMatrix;

	float4 c_inpos = c_VP * c_M * make_float4(l_pos, 1);//clipspace
	float4 p_inpos = p_VP * p_M * make_float4(l_pos, 1);//clipspace

	float3 c_ndc = make_float3(c_inpos) / c_inpos.w;
	float3 p_ndc = make_float3(p_inpos) / p_inpos.w;

	float2 c_uv = (make_float2(c_ndc) + 1.f) / 2.f;//0->1
	float2 p_uv = (make_float2(p_ndc) + 1.f) / 2.f;

	float2 vel = c_uv - p_uv;

	float3 velcol = make_float3(0);

	velcol = make_float3(vel, 0);

	texWrite(make_float4(velcol, 1), t_globals.FrameBuffer.velocity_surfobject, t_current_pix);
}

__global__ void computePrimaryVisibility(const IntegratorGlobals t_globals) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------
	uint32_t seed = current_pix.x + current_pix.y * frame_res.x;
	seed *= t_globals.FrameIndex;

	float2 ndc_uv = screen_uv * 2 - 1;
	Ray primary_ray = t_globals.SceneDescriptor.ActiveCamera->generateRay(frame_res.x, frame_res.y, ndc_uv);

	ShapeIntersection payload = IntegratorPipeline::Intersect(t_globals, primary_ray);

	recordGBufferAny(t_globals, current_pix, payload);

	if (payload.hit_distance < 0) {
		recordGBufferMiss(t_globals, current_pix);
	}
	else {
		recordGBufferHit(t_globals, current_pix, payload);
	}

	computeVelocity(t_globals, current_pix);

	//out seeds---------
	texWrite(make_float4(make_float3(seed), 1), t_globals.FrameBuffer.seeds_surfobject, current_pix);
}

__global__ void composeCompositeImage(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
		current_pix);
	RGBSpectrum sampled_radiance = RGBSpectrum(sampled_irradiance);

	float4 sampled_albedo = texReadNearest(t_globals.FrameBuffer.albedo_surfobject, current_pix);

	sampled_radiance *= RGBSpectrum(sampled_albedo);//MODULATE assume BRDF normalised

	RGBSpectrum frag_spectrum = sampled_radiance;
	//EOTF
	frag_spectrum = gammaCorrection(frag_spectrum);
	frag_spectrum = toneMapping(frag_spectrum, t_globals.SceneDescriptor.ActiveCamera->exposure);

	float4 frag_color = make_float4(frag_spectrum, 1);

	texWrite(frag_color, t_globals.FrameBuffer.composite_surfobject, current_pix);
}