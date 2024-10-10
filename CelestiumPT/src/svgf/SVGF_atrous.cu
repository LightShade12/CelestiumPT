#include "SVGFPasses.cuh"

#include "integrator.cuh"
#include "spectrum.cuh"
#include "storage.cuh"
#include "maths/linear_algebra.cuh"
#include "film.cuh"
#include "error_check.cuh"
#include "SVGFEdgeStoppingFunctions.cuh"
#include "cuda_utility.cuh"

#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float4 spatialVarianceEstimate(const IntegratorGlobals& globals, int2 t_current_pix) {
	float3 sampled_normal = normalize(make_float3(texReadNearest(globals.FrameBuffer.world_normals_render_surface_object,
		t_current_pix)));
	float sampled_depth = texReadNearest(globals.FrameBuffer.depth_render_surface_object,
		t_current_pix).x;
	float histlen = texReadNearest(globals.FrameBuffer.current_moments_render_surface_object, t_current_pix).w;

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(globals.FrameBuffer.depth_render_surface_object, t_current_pix, globals.FrameBuffer.resolution).x,
		dFdy(globals.FrameBuffer.depth_render_surface_object, t_current_pix, globals.FrameBuffer.resolution).x);

	float weight_sum = 0.f;
	float2 f_moments = make_float2(0.f);
	float3 f_irradiance = make_float3(0);

	int radius = 3; // 7x7 Gaussian Kernel

	//TODO:skip centre pixel?
	for (int yy = -radius; yy <= radius; ++yy)
	{
		for (int xx = -radius; xx <= radius; ++xx)
		{
			int2 offset = make_int2(xx, yy);
			int2 tap_pix = t_current_pix + offset;
			tap_pix = clamp(tap_pix, make_int2(0, 0), (globals.FrameBuffer.resolution - 1));

			float4 tap_irradiance = texReadNearest(globals.FrameBuffer.current_irradiance_render_surface_object,
				tap_pix);
			float3 tap_normal = make_float3(texReadNearest(globals.FrameBuffer.world_normals_render_surface_object,
				tap_pix));
			float tap_depth = texReadNearest(globals.FrameBuffer.depth_render_surface_object,
				tap_pix).x;

			float l = getLuminance(RGBSpectrum(tap_irradiance));

			float nw = normalWeight((sampled_normal), normalize(tap_normal));
			float dw = depthWeight(sampled_depth, tap_depth, dgrad, make_float2(offset));

			//compute luminance weights?

			float w = dw * nw;
			if (isnan(w))w = 0;

			f_irradiance += make_float3(tap_irradiance) * w;
			f_moments += make_float2(l, l * l) * w;//possibly sample moments?
			weight_sum += w;
		}
	}

	weight_sum = fmaxf(weight_sum, 1e-6f);
	f_moments /= weight_sum;
	f_irradiance /= weight_sum;

	float variance = fabsf(f_moments.y - Sqr(f_moments.x));
	variance *= fmaxf(4.f / histlen, 1.f);//boost for 1st few frames

	return make_float4(f_irradiance, variance);
}

//updates filtered irradiance and filtered variance
__global__ void SVGFPass(const IntegratorGlobals globals, int stepsize) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;
	//----------------------------------------------

	int current_objID = texReadNearest(globals.FrameBuffer.objectID_render_surface_object, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no filter
		float sampled_variance = texReadNearest(globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix).x;
		float4 sampled_irradiance = texReadNearest(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		//out----
		texWrite(sampled_irradiance,
			globals.FrameBuffer.filtered_irradiance_back_render_surface_object,
			current_pix);
		texWrite(make_float4(make_float3(sampled_variance), 1),
			globals.FrameBuffer.filtered_variance_render_back_surfobj,
			current_pix);
		return;
	}

	float4 sampled_irradiance = texReadNearest(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
	float3 sampled_normal = normalize(make_float3(texReadNearest(globals.FrameBuffer.world_normals_render_surface_object,
		current_pix)));
	float sampled_depth = texReadNearest(globals.FrameBuffer.depth_render_surface_object,
		current_pix).x;
	//float sampled_variance = texReadNearest(globals.FrameBuffer.filtered_variance_render_front_surfobj,
	//	current_pix).x;
	float sampled_variance = texReadGaussianWeighted(globals.FrameBuffer.filtered_variance_render_front_surfobj, frameres,
		current_pix);

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(globals.FrameBuffer.depth_render_surface_object, current_pix, globals.FrameBuffer.resolution).x,
		dFdy(globals.FrameBuffer.depth_render_surface_object, current_pix, globals.FrameBuffer.resolution).x);

	bool use_5x5filter = globals.IntegratorCFG.use_5x5_filter;

	const constexpr float filterKernel3x3[] =
	{
		0.0625, 0.125, 0.0625,
		0.125, 0.25, 0.125,
		0.0625, 0.125, 0.0625 };

	const constexpr float filterKernel5x5[] =
	{
		0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
		0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
		0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
		0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
		0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625
	};

	int radius = (use_5x5filter) ? 2 : 1;

	float4 f_irradiance = sampled_irradiance;
	float f_variance = sampled_variance;
	float wsum = 1.0;

	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++) {
			if (x == 0 && y == 0)continue;
			int2 offset = make_int2(x, y) * stepsize;
			int2 tap_pix = current_pix + offset;
			tap_pix = clamp(tap_pix, make_int2(0, 0), (globals.FrameBuffer.resolution - 1));

			float4 tap_irradiance = texReadNearest(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
				tap_pix);
			float3 tap_normal = make_float3(texReadNearest(globals.FrameBuffer.world_normals_render_surface_object,
				tap_pix));
			float tap_depth = texReadNearest(globals.FrameBuffer.depth_render_surface_object,
				tap_pix).x;
			float tap_variance = texReadNearest(globals.FrameBuffer.filtered_variance_render_front_surfobj,
				tap_pix).x;

			float nw = normalWeight((sampled_normal), normalize(tap_normal));
			float dw = depthWeight(sampled_depth, tap_depth, dgrad, make_float2(offset));
			float lw = 1;
			lw = luminanceWeight(
				getLuminance(RGBSpectrum(sampled_irradiance)),
				getLuminance(RGBSpectrum(tap_irradiance)), sampled_variance);

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
	//f_variance = sampled_variance;
	//out----
	texWrite((f_irradiance),
		globals.FrameBuffer.filtered_irradiance_back_render_surface_object,
		current_pix);
	texWrite(make_float4(make_float3(f_variance), 1),
		globals.FrameBuffer.filtered_variance_render_back_surfobj,
		current_pix);
}