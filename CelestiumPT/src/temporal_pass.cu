#include "temporal_pass.cuh"
#include "cuda_utility.cuh"
#include "scene_geometry.cuh"
#include "device_scene.cuh"
#include "maths/linear_algebra.cuh"
#include <device_launch_parameters.h>

//used for pure TAA samples
__global__ void temporalAccumulate(const IntegratorGlobals globals) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texReadNearest(globals.FrameBuffer.current_irradiance_render_surface_object, current_pix);
	float4 sampled_moments = texReadNearest(globals.FrameBuffer.current_moments_render_surface_object, current_pix);

	RGBSpectrum final_irradiance = RGBSpectrum(sampled_irradiance);
	float2 final_moments = make_float2(sampled_moments.x, sampled_moments.y);

	int current_objID = texReadNearest(globals.FrameBuffer.objectID_render_surface_object, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no accumulate

		//feedback--
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			current_pix);
		//out---
		//texWrite(make_float4(make_float3(0), 0),
		//	globals.FrameBuffer.filtered_variance_render_front_surfobj,
		//	current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		return;
	}

	float4 current_vel = texReadNearest(globals.FrameBuffer.velocity_render_surface_object, current_pix);

	//reproject
	int2 res = globals.FrameBuffer.resolution;
	float2 velf2 = make_float2(current_vel.x, current_vel.y);
	float2 pixel_offset = (velf2)*make_float2(res);

	int2 prev_px = current_pix - make_int2(pixel_offset);
	float2 prev_pxf = make_float2(current_pix) - pixel_offset;

	//new fragment; out of screen
	if (prev_px.x < 0 || prev_px.x >= res.x ||
		prev_px.y < 0 || prev_px.y >= res.y) {
		//feedback--
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			current_pix);
		//out---
		//float var = spatialVarianceEstimate(globals, current_pix);
		//texWrite(make_float4(make_float3(var), 0),
		//	globals.FrameBuffer.filtered_variance_render_front_surfobj,
		//	current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);

		return;
	}

	bool prj_success = !rejectionHeuristic(globals, prev_px, current_pix);

	//disocclusion/ reproj failure
	if (!prj_success) {
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			current_pix);
		//out---
		//float var = spatialVarianceEstimate(globals, current_pix);
		//texWrite(make_float4(make_float3(var), 0),
		//	globals.FrameBuffer.filtered_variance_render_front_surfobj,
		//	current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		return;
	}

	float4 hist_irradiance = texReadBilinear(globals.FrameBuffer.history_integrated_irradiance_front_surfobj, prev_pxf,
		frameres, false);
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

	//feedback: irradiance
	texWrite(make_float4(final_irradiance, irradiance_hist_len + 1),
		globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
		current_pix);

	//out----
	//float variance;
	//if (moments_hist_len < 4) {
	//	variance = spatialVarianceEstimate(globals, current_pix);
	//}
	//else {
	//	float2 final_v = final_moments;
	//	variance = fabsf(final_v.y - (Sqr(final_v.x)));
	//}
	//texWrite(make_float4(make_float3(variance), 1),
	//	globals.FrameBuffer.filtered_variance_render_front_surfobj,
	//	current_pix);
	texWrite(make_float4(final_irradiance, irradiance_hist_len + 1),
		globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
}

__device__ bool rejectionHeuristic(const IntegratorGlobals& globals, int2 prev_pix, int2 cur_px)
{
	float4 c_lpos_sample = surf2Dread<float4>(globals.FrameBuffer.local_positions_render_surface_object,
		cur_px.x * (int)sizeof(float4), cur_px.y);
	float3 c_lpos = make_float3(c_lpos_sample);

	float4 c_objID_sample = surf2Dread<float4>(globals.FrameBuffer.objectID_render_surface_object,
		cur_px.x * (int)sizeof(float4), cur_px.y);
	int c_objID = c_objID_sample.x;

	Mat4 p_model = globals.SceneDescriptor.device_geometry_aggregate->DeviceMeshesBuffer[c_objID].prev_modelMatrix;

	//DEPTH HEURISTIC-------------
	float4 p_depth_sample = surf2Dread<float4>(globals.FrameBuffer.history_depth_render_surface_object,
		prev_pix.x * (int)sizeof(float4), prev_pix.y);
	float p_depth = p_depth_sample.x;

	float3 p_cpos = make_float3(globals.SceneDescriptor.active_camera->prev_viewMatrix.inverse() * make_float4(0, 0, 0, 1));
	float3 p_wpos = make_float3(p_model * make_float4(c_lpos, 1));//clipspace

	float estimated_p_depth = length(p_cpos - p_wpos);

	float TEMPORAL_DEPTH_REJECT_THRESHOLD = 0.045f;

	if (fabsf(estimated_p_depth - p_depth) > (p_depth * TEMPORAL_DEPTH_REJECT_THRESHOLD)) {
		return true;
	}
	return false;

	//NORMALS HEURISTIC------------
	float4 p_wnorm_sample = surf2Dread<float4>(globals.FrameBuffer.history_world_normals_render_surface_object,
		prev_pix.x * (int)sizeof(float4), prev_pix.y);
	float3 p_wnorm = normalize(make_float3(p_wnorm_sample));

	float4 c_lnorm_sample = surf2Dread<float4>(globals.FrameBuffer.local_normals_render_surface_object,
		prev_pix.x * (int)sizeof(float4), prev_pix.y);
	float3 c_lnorm = make_float3(c_lnorm_sample);

	float3 estimated_p_wnorm = normalize(make_float3(p_model * make_float4(c_lnorm, 0)));

	float TEMPORAL_NORMALS_REJECT_THRESHOLD = fabsf(cosf(deg2rad(45)));//TODO:make consexpr

	if (AbsDot(p_wnorm, estimated_p_wnorm) < TEMPORAL_NORMALS_REJECT_THRESHOLD) {
		return true;
	}

	return false;
}