#pragma once
#include "storage.cuh"
#include "cuda_utility.cuh"
#include "film.cuh"
#include "device_camera.cuh"

#include "ray.cuh"
#include "shape_intersection.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

__global__ void computePrimaryVisibility(const IntegratorGlobals t_globals) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	screen_uv = screen_uv * 2 - 1;
	Ray primary_ray = t_globals.SceneDescriptor.ActiveCamera->generateRay(frame_res.x, frame_res.y, screen_uv);

	ShapeIntersection payload = IntegratorPipeline::Intersect(t_globals, primary_ray);

	recordGBufferAny(t_globals, current_pix, payload);

	if (payload.hit_distance < 0) {
		recordGBufferMiss(t_globals, current_pix);
		return;
	}
	recordGBufferHit(t_globals, current_pix, payload);
}