#include "storage.cuh"
#include "maths/linear_algebra.cuh"
#include "cuda_utility.cuh"
#include "svgf_weight_functions.cuh"

#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ constexpr int SPATIAL_ESTIMATE_MIN_FRAMES = 4;

__device__ float4 spatialVarianceEstimate(const IntegratorGlobals& globals, int2 t_current_pix) {
	float3 sampled_normal = normalize(make_float3(texReadNearest(globals.FrameBuffer.world_normals_surfobject,
		t_current_pix)));
	float sampled_depth = texReadNearest(globals.FrameBuffer.depth_surfobject,
		t_current_pix).x;
	float hist_len = texReadNearest(globals.FrameBuffer.integrated_moments_front_surfobject, t_current_pix).w;

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(globals.FrameBuffer.depth_surfobject, t_current_pix, globals.FrameBuffer.resolution).x,
		dFdy(globals.FrameBuffer.depth_surfobject, t_current_pix, globals.FrameBuffer.resolution).x);

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

			float4 tap_irradiance = texReadNearest(globals.FrameBuffer.raw_irradiance_surfobject,
				tap_pix);
			float3 tap_normal = make_float3(texReadNearest(globals.FrameBuffer.world_normals_surfobject,
				tap_pix));
			float tap_depth = texReadNearest(globals.FrameBuffer.depth_surfobject,
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
	//variance *= SPATIAL_ESTIMATE_MIN_FRAMES / fminf(hist_len, SPATIAL_ESTIMATE_MIN_FRAMES);//boost for 1st few frames; NaN bugged

	return make_float4(f_irradiance, variance);
}

__global__ void estimateVariance(const IntegratorGlobals t_globals)
{
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
		texWrite(make_float4(make_float3(0), 0),
			t_globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
			current_pix);
		return;
	}

	float4 moments = texReadNearest(t_globals.FrameBuffer.integrated_moments_front_surfobject, current_pix);
	float hist_len = moments.w;

	float variance = 0;
	if (hist_len < SPATIAL_ESTIMATE_MIN_FRAMES && t_globals.IntegratorCFG.svgf_enabled) {
		float4 spatial_estimate = spatialVarianceEstimate(t_globals, current_pix);//var_irr
		variance = spatial_estimate.w;

		//assuming mom_hist==irr_hist
		texWrite(make_float4(make_float3(spatial_estimate), hist_len),
			t_globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
			current_pix);
	}
	else {
		variance = fabsf(moments.y - (Sqr(moments.x)));
	}

	texWrite(make_float4(make_float3(variance), 1),
		t_globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
		current_pix);
}