#include "denoiser.cuh"
#include "cuda_utility.cuh"
#include "samplers.cuh"
#include "maths/constants.cuh"

__device__ int2 randomPosStratum(uint32_t seed) {
	float2 rand_offset = Samplers::get2D_PCGHash(seed);  // Already in [0, 1] range

	int2 pos;
	pos.x = int(rand_offset.x * ASVGF_STRATUM_SIZE);
	pos.y = int(rand_offset.y * ASVGF_STRATUM_SIZE);

	return pos;
}

__global__ void mergeSamples(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x / ASVGF_STRATUM_SIZE) || (current_pix.y >= frame_res.y / ASVGF_STRATUM_SIZE)) return;
	//=========================================================

	const int2 grad_pix = current_pix;

	//sample selection------------------
	uint32_t seed = grad_pix.x + grad_pix.y * frame_res.x;
	seed *= t_globals.FrameIndex + 2;
	//int2 pos_in_stratum = make_int2(1);//hardcoded centre selection
	int2 pos_in_stratum = randomPosStratum(seed);
	//store stratum pos
	texWrite(make_float4(make_float3(packStratumPos(pos_in_stratum)), 1),
		t_globals.FrameBuffer.asvgf_gradient_sample_surfobject,
		grad_pix);

	int2 sampling_pix = grad_pix * ASVGF_STRATUM_SIZE + pos_in_stratum;
	sampling_pix = clamp(sampling_pix, { 0,0 }, frame_res - 1);

	int current_objID = texReadNearest(t_globals.FrameBuffer.objectID_surfobject, sampling_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0)
	{
		float4 seed = texReadNearest(t_globals.FrameBuffer.seeds_surfobject, sampling_pix);
		seed.w = -1;//mark as invalid gradient sample
		texWrite(seed, t_globals.FrameBuffer.seeds_surfobject, sampling_pix);
		return;
	}

	//reproject
	float4 sampled_velocity = texReadNearest(t_globals.FrameBuffer.velocity_surfobject, sampling_pix);
	float2 current_velocity = make_float2(sampled_velocity.x, sampled_velocity.y);
	float2 pixel_offset = current_velocity * make_float2(frame_res);

	int2 prev_px = sampling_pix - make_int2(pixel_offset);
	float2 prev_pxf = make_float2(sampling_pix) - pixel_offset;

	//new fragment; out of screen
	if (prev_px.x < 0 || prev_px.x >= frame_res.x ||
		prev_px.y < 0 || prev_px.y >= frame_res.y)
	{
		float4 seed = texReadNearest(t_globals.FrameBuffer.seeds_surfobject, sampling_pix);
		seed.w = -1;//mark as invalid gradient sample
		texWrite(seed, t_globals.FrameBuffer.seeds_surfobject, sampling_pix);
		return;
	}

	bool reproj_success = !rejectionHeuristic(t_globals, prev_px, sampling_pix);

	if (!reproj_success)
	{
		float4 seed = texReadNearest(t_globals.FrameBuffer.seeds_surfobject, sampling_pix);
		seed.w = -1;//mark as invalid gradient sample
		texWrite(seed, t_globals.FrameBuffer.seeds_surfobject, sampling_pix);
		return;
	}

	//reproj old shading
	float4 old_shading = texReadNearest(t_globals.FrameBuffer.raw_irradiance_surfobject, prev_px);
	texWrite(old_shading, t_globals.FrameBuffer.history_shading_surfobject, sampling_pix);

	//reproj seeds
	float4 old_seed = texReadNearest(t_globals.FrameBuffer.history_seeds_surfobject, prev_px);
	old_seed.w = 1;//reset grad validation flags
	texWrite(old_seed, t_globals.FrameBuffer.seeds_surfobject, sampling_pix);

	//reproj surface samples
	// G-buffer
	float4 old_w_norm = texReadNearest(t_globals.FrameBuffer.history_world_normals_surfobject, prev_px);
	float4 old_l_norm = texReadNearest(t_globals.FrameBuffer.history_local_normals_surfobject, prev_px);

	float4 old_w_pos = texReadNearest(t_globals.FrameBuffer.history_world_positions_surfobject, prev_px);
	float4 old_l_pos = texReadNearest(t_globals.FrameBuffer.history_local_positions_surfobject, prev_px);

	float4 old_depth = texReadNearest(t_globals.FrameBuffer.history_depth_surfobject, prev_px);
	float4 old_viewdir = texReadNearest(t_globals.FrameBuffer.history_viewdirections_surfobject, prev_px);

	float4 old_triangle_ID = texReadNearest(t_globals.FrameBuffer.history_triangleID_surfobject, prev_px);
	float4 old_object_ID = texReadNearest(t_globals.FrameBuffer.history_objectID_surfobject, prev_px);

	float4 old_uv = texReadNearest(t_globals.FrameBuffer.history_UVs_surfobject, prev_px);
	float4 old_bary = texReadNearest(t_globals.FrameBuffer.history_bary_surfobject, prev_px);
	//-------
	texWrite(old_w_norm, t_globals.FrameBuffer.world_normals_surfobject, sampling_pix);
	texWrite(old_l_norm, t_globals.FrameBuffer.local_normals_surfobject, sampling_pix);

	texWrite(old_w_pos, t_globals.FrameBuffer.world_positions_surfobject, sampling_pix);
	texWrite(old_l_pos, t_globals.FrameBuffer.local_positions_surfobject, sampling_pix);

	texWrite(old_depth, t_globals.FrameBuffer.depth_surfobject, sampling_pix);
	texWrite(old_viewdir, t_globals.FrameBuffer.viewdirections_surfobject, sampling_pix);

	texWrite(old_triangle_ID, t_globals.FrameBuffer.triangleID_surfobject, sampling_pix);
	texWrite(old_object_ID, t_globals.FrameBuffer.objectID_surfobject, sampling_pix);

	texWrite(old_uv, t_globals.FrameBuffer.UVs_surfobject, sampling_pix);
	texWrite(old_bary, t_globals.FrameBuffer.bary_surfobject, sampling_pix);
}