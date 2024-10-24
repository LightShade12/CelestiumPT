#include "svgf_temporal_reprojection.cuh"

#include "svgf_passes.cuh"
#include "cuda_utility.cuh"
#include "temporal_pass.cuh"

#include "maths/linear_algebra.cuh"
#include <device_launch_parameters.h>

__global__ void temporalIntegrate(const IntegratorGlobals globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texReadNearest(globals.FrameBuffer.current_irradiance_render_surface_object, current_pix);
	float4 sampled_moments = texReadNearest(globals.FrameBuffer.current_moments_render_surface_object, current_pix);

	RGBSpectrum final_irradiance = RGBSpectrum(sampled_irradiance);
	float2 final_moments = make_float2(sampled_moments.x, sampled_moments.y);

	int current_objID = texReadNearest(globals.FrameBuffer.objectID_render_surface_object, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no accumulate

		//feedback
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);

		//out---
		texWrite(make_float4(make_float3(0), 0),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		return;
	}

	float4 sampled_velocity = texReadNearest(globals.FrameBuffer.velocity_render_surface_object, current_pix);

	//reproject
	float2 current_velocity = make_float2(sampled_velocity.x, sampled_velocity.y);
	float2 pixel_offset = current_velocity * make_float2(frame_res);

	int2 prev_px = current_pix - make_int2(pixel_offset);
	float2 prev_pxf = make_float2(current_pix) - pixel_offset;

	//new fragment; out of screen
	if (prev_px.x < 0 || prev_px.x >= frame_res.x ||
		prev_px.y < 0 || prev_px.y >= frame_res.y)
	{
		//feedback
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);

		//out---
		float4 irr_var;
		irr_var = make_float4(final_irradiance);
		irr_var.w = fabsf(final_moments.y - (Sqr(final_moments.x)));

		if (globals.IntegratorCFG.svgf_enabled)
			irr_var = spatialVarianceEstimate(globals, current_pix);

		texWrite(make_float4(make_float3(irr_var.w), 0),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(make_float3(irr_var), 0),/*final_irradiance*/
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);

		return;
	}

	bool prj_success = !rejectionHeuristic(globals, prev_px, current_pix);

	//disocclusion/ reproj failure
	if (!prj_success)
	{
		//feedback
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);

		//out---
		float4 irr_var;
		irr_var = make_float4(final_irradiance);
		irr_var.w = fabsf(final_moments.y - (Sqr(final_moments.x)));

		if (globals.IntegratorCFG.svgf_enabled)
			irr_var = spatialVarianceEstimate(globals, current_pix);
		texWrite(make_float4(make_float3(irr_var.w), 0),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(make_float3(irr_var), 0),/*final_irradiance*/
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		return;
	}

	float4 hist_irradiance = texReadBilinear(globals.FrameBuffer.history_integrated_irradiance_front_surfobj, prev_pxf,
		frame_res, false);
	float4 hist_moments = texReadNearest(globals.FrameBuffer.history_integrated_moments_front_surfobj, prev_px);

	float irradiance_hist_len = hist_irradiance.w;
	float moments_hist_len = hist_moments.w;

	const constexpr int MAX_ACCUMULATION_FRAMES = 16;

	final_irradiance = RGBSpectrum(
		lerp(make_float3(hist_irradiance), make_float3(final_irradiance),
			1.f / fminf(float(irradiance_hist_len + 1), MAX_ACCUMULATION_FRAMES))
	);

	final_moments = lerp(make_float2(hist_moments.x, hist_moments.y), final_moments,
		1.f / fminf(float(moments_hist_len + 1), MAX_ACCUMULATION_FRAMES));

	//feedback: moments
	texWrite(make_float4(final_moments.x, final_moments.y, 0, moments_hist_len + 1),
		globals.FrameBuffer.history_integrated_moments_back_surfobj,
		current_pix);

	float4 variance;
	if (moments_hist_len < 4 && globals.IntegratorCFG.svgf_enabled) {
		variance = spatialVarianceEstimate(globals, current_pix);//var_irr
		final_irradiance = RGBSpectrum(variance);//averaged irradiance in variance var
	}
	else {
		variance.w = fabsf(final_moments.y - (Sqr(final_moments.x)));
	}

	texWrite(make_float4(make_float3(variance.w), 1),
		globals.FrameBuffer.filtered_variance_render_front_surfobj,
		current_pix);
	//out----
	texWrite(make_float4(final_irradiance, irradiance_hist_len + 1),//send out hist_len to restore it after 1st filterpass
		globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
}