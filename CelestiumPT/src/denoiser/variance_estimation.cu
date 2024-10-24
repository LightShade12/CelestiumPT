#include "storage.cuh"
#include "maths/linear_algebra.cuh"
#include "cuda_utility.cuh"
#include "svgf_weight_functions.cuh"

#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float4 spatialVarianceEstimate(const IntegratorGlobals& globals, int2 t_current_pix) {
	float3 sampled_normal = normalize(make_float3(texReadNearest(globals.FrameBuffer.world_normals_surfobject,
		t_current_pix)));
	float sampled_depth = texReadNearest(globals.FrameBuffer.depth_surfobject,
		t_current_pix).x;
	float histlen = texReadNearest(globals.FrameBuffer.raw_moments_surfobject, t_current_pix).w;

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
	variance *= fmaxf(4.f / histlen, 1.f);//boost for 1st few frames

	return make_float4(f_irradiance, variance);
}