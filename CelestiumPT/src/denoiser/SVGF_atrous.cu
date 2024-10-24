#include "denoiser.cuh"

#include "storage.cuh"
#include "maths/linear_algebra.cuh"

#include "svgf_weight_functions.cuh"
#include "cuda_utility.cuh"

#define __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ constexpr float filterKernel3x3[] =
{
	0.0625, 0.125, 0.0625,
	0.125, 0.25, 0.125,
	0.0625, 0.125, 0.0625 
};

__constant__ constexpr float filterKernel5x5[] =
{
	0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
	0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
	0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
	0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
	0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625
};

//updates filtered irradiance and filtered variance
__global__ void atrousSVGF(const IntegratorGlobals t_globals, int t_stepsize) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//=========================================================

	int current_objID = texReadNearest(t_globals.FrameBuffer.objectID_surfobject, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no filter
		float sampled_variance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
			current_pix).x;
		float4 sampled_irradiance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
			current_pix);
		//out----
		texWrite(sampled_irradiance,
			t_globals.FrameBuffer.svgf_filtered_irradiance_back_surfobject,
			current_pix);
		texWrite(make_float4(make_float3(sampled_variance), 1),
			t_globals.FrameBuffer.svgf_filtered_variance_back_surfobject,
			current_pix);
		return;
	}

	float4 sampled_irradiance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
		current_pix);
	float3 sampled_normal = normalize(make_float3(texReadNearest(t_globals.FrameBuffer.world_normals_surfobject,
		current_pix)));
	float sampled_depth = texReadNearest(t_globals.FrameBuffer.depth_surfobject,
		current_pix).x;
	float sampled_variance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
		current_pix).x;
	float sampled_filtered_variance = texReadGaussianWeighted(t_globals.FrameBuffer.svgf_filtered_variance_front_surfobject, frame_res,
		current_pix);

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(t_globals.FrameBuffer.depth_surfobject, current_pix, t_globals.FrameBuffer.resolution).x,
		dFdy(t_globals.FrameBuffer.depth_surfobject, current_pix, t_globals.FrameBuffer.resolution).x);

	bool use_5x5filter = t_globals.IntegratorCFG.use_5x5_filter;

	int radius = (use_5x5filter) ? 2 : 1;

	float4 f_irradiance = sampled_irradiance;
	float f_variance = sampled_variance;
	float wsum = 1.0;

	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++) {
			if (x == 0 && y == 0)continue;
			int2 offset = make_int2(x, y) * t_stepsize;
			int2 tap_pix = current_pix + offset;
			tap_pix = clamp(tap_pix, make_int2(0, 0), (frame_res - 1));

			float4 tap_irradiance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
				tap_pix);
			float tap_variance = texReadNearest(t_globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
				tap_pix).x;

			float3 tap_normal = make_float3(texReadNearest(t_globals.FrameBuffer.world_normals_surfobject,
				tap_pix));
			float tap_depth = texReadNearest(t_globals.FrameBuffer.depth_surfobject,
				tap_pix).x;

			float nw = normalWeight((sampled_normal), normalize(tap_normal));
			float dw = depthWeight(sampled_depth, tap_depth, dgrad, make_float2(offset));
			float lw = 1;
			lw = luminanceWeight(
				getLuminance(RGBSpectrum(sampled_irradiance)),
				getLuminance(RGBSpectrum(tap_irradiance)), sampled_filtered_variance);

			float w = (dw * nw * lw);

			// scale by the filtering kernel
			float h = ((use_5x5filter) ? filterKernel5x5 : filterKernel3x3)[(x + radius) + (y + radius) * ((2 * radius) + 1)];
			float hw = h * w;

			// add to total irradiance
			f_irradiance += tap_irradiance * hw;
			f_variance += Sqr(hw) * tap_variance;
			wsum += hw;
		}
	}

	f_irradiance /= wsum;
	f_variance /= Sqr(wsum);

	f_irradiance.w = sampled_irradiance.w;//restore history length for temporal feedback

	//out----
	texWrite((f_irradiance),
		t_globals.FrameBuffer.svgf_filtered_irradiance_back_surfobject,
		current_pix);
	texWrite(make_float4(make_float3(f_variance), 1),
		t_globals.FrameBuffer.svgf_filtered_variance_back_surfobject,
		current_pix);
}