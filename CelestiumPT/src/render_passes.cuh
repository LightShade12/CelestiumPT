#pragma once
#include "storage.cuh"
#include "cuda_utility.cuh"
#include "film.cuh"
#include "device_camera.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void composeCompositeImage(const IntegratorGlobals globals) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texReadNearest(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
	RGBSpectrum sampled_radiance = RGBSpectrum(sampled_irradiance);

	float4 sampled_albedo = texReadNearest(globals.FrameBuffer.albedo_render_surface_object, current_pix);

	sampled_radiance *= RGBSpectrum(sampled_albedo);//MODULATE assume BRDF normalised

	RGBSpectrum frag_spectrum = sampled_radiance;
	//EOTF
	frag_spectrum = gammaCorrection(frag_spectrum);
	frag_spectrum = toneMapping(frag_spectrum, globals.SceneDescriptor.ActiveCamera->exposure);

	float4 frag_color = make_float4(frag_spectrum, 1);

	texWrite(frag_color, globals.FrameBuffer.composite_render_surface_object, current_pix);
}