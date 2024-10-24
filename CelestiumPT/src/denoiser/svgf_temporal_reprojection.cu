#include "svgf_temporal_reprojection.cuh"

#include "scene_geometry.cuh"
#include "device_scene.cuh"

#include "svgf_passes.cuh"
#include "cuda_utility.cuh"

#include "maths/linear_algebra.cuh"
#include <device_launch_parameters.h>

//depth2 is sampled depth
__device__ bool testReprojectedDepth(float depth1, float depth2) {
	float TEMPORAL_DEPTH_REJECT_THRESHOLD = 0.045f;

	if (fabsf(depth1 - depth2) > (depth2 * TEMPORAL_DEPTH_REJECT_THRESHOLD)) {
		return true;
	}
	return false;
}

__device__ bool testReprojectedNormals(float3 n1, float3 n2)
{
	float TEMPORAL_NORMALS_REJECT_THRESHOLD = fabsf(cosf(deg2rad(45)));//TODO:make consexpr

	if (AbsDot(n1, n2) < TEMPORAL_NORMALS_REJECT_THRESHOLD) {
		return true;
	}

	return false;
}

__device__ bool rejectionHeuristic(const IntegratorGlobals& t_globals, int2 t_prev_pix, int2 t_current_pix)
{
	float4 c_lpos_sample = texReadNearest(t_globals.FrameBuffer.local_positions_surfobject,
		t_current_pix);
	float3 c_lpos = make_float3(c_lpos_sample);

	float4 c_objID_sample = texReadNearest(t_globals.FrameBuffer.objectID_surfobject,
		t_current_pix);
	int c_objID = c_objID_sample.x;

	Mat4 p_model = t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMeshesBuffer[c_objID].prev_modelMatrix;

	//DEPTH HEURISTIC-------------
	float4 p_depth_sample = texReadNearest(t_globals.FrameBuffer.history_depth_surfobject,
		t_prev_pix);
	float p_depth = p_depth_sample.x;

	float3 p_cpos = make_float3(t_globals.SceneDescriptor.ActiveCamera->prev_viewMatrix.inverse() * make_float4(0, 0, 0, 1));
	float3 p_wpos = make_float3(p_model * make_float4(c_lpos, 1));//clipspace

	float estimated_p_depth = length(p_cpos - p_wpos);

	return testReprojectedDepth(estimated_p_depth, p_depth);

	//NORMALS HEURISTIC------------
	float4 p_wnorm_sample = texReadNearest(t_globals.FrameBuffer.history_world_normals_surfobject,
		t_prev_pix);
	float3 p_wnorm = normalize(make_float3(p_wnorm_sample));

	float4 c_lnorm_sample = texReadNearest(t_globals.FrameBuffer.local_normals_surfobject,
		t_prev_pix);
	float3 c_lnorm = make_float3(c_lnorm_sample);

	float3 estimated_p_wnorm = normalize(make_float3(p_model * make_float4(c_lnorm, 0)));

	return testReprojectedNormals(p_wnorm, estimated_p_wnorm);
}

__global__ void temporalAccumulate(const IntegratorGlobals globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texReadNearest(globals.FrameBuffer.raw_irradiance_surfobject, current_pix);
	float4 sampled_moments = texReadNearest(globals.FrameBuffer.raw_moments_surfobject, current_pix);

	RGBSpectrum final_irradiance = RGBSpectrum(sampled_irradiance);
	float2 final_moments = make_float2(sampled_moments.x, sampled_moments.y);

	int current_objID = texReadNearest(globals.FrameBuffer.objectID_surfobject, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no accumulate

		//feedback
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.integrated_moments_back_surfobject,
			current_pix);

		//out---
		texWrite(make_float4(make_float3(0), 0),
			globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
			current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
			current_pix);
		return;
	}

	float4 sampled_velocity = texReadNearest(globals.FrameBuffer.velocity_surfobject, current_pix);

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
			globals.FrameBuffer.integrated_moments_back_surfobject,
			current_pix);

		//out---
		float4 irr_var;
		irr_var = make_float4(final_irradiance);
		irr_var.w = fabsf(final_moments.y - (Sqr(final_moments.x)));

		if (globals.IntegratorCFG.svgf_enabled)
			irr_var = spatialVarianceEstimate(globals, current_pix);

		texWrite(make_float4(make_float3(irr_var.w), 0),
			globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
			current_pix);
		texWrite(make_float4(make_float3(irr_var), 0),/*final_irradiance*/
			globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
			current_pix);

		return;
	}

	bool prj_success = !rejectionHeuristic(globals, prev_px, current_pix);

	//disocclusion/ reproj failure
	if (!prj_success)
	{
		//feedback
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.integrated_moments_back_surfobject,
			current_pix);

		//out---
		float4 irr_var;
		irr_var = make_float4(final_irradiance);
		irr_var.w = fabsf(final_moments.y - (Sqr(final_moments.x)));

		if (globals.IntegratorCFG.svgf_enabled)
			irr_var = spatialVarianceEstimate(globals, current_pix);
		texWrite(make_float4(make_float3(irr_var.w), 0),
			globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
			current_pix);
		texWrite(make_float4(make_float3(irr_var), 0),/*final_irradiance*/
			globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
			current_pix);
		return;
	}

	float4 hist_irradiance = texReadBilinear(globals.FrameBuffer.integrated_irradiance_front_surfobject, prev_pxf,
		frame_res, false);
	float4 hist_moments = texReadNearest(globals.FrameBuffer.integrated_moments_front_surfobject, prev_px);

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
		globals.FrameBuffer.integrated_moments_back_surfobject,
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
		globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
		current_pix);
	//out----
	texWrite(make_float4(final_irradiance, irradiance_hist_len + 1),//send out hist_len to restore it after 1st filterpass
		globals.FrameBuffer.svgf_filtered_irradiance_front_surfobject,
		current_pix);
}