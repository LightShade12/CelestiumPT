#include "denoiser.cuh"
#include "storage.cuh"
#include "maths/constants.cuh"
#include "cuda_utility.cuh"
#include "svgf_weight_functions.cuh"

__constant__ constexpr float box_kernel[9] = {
	1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
	1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
	1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f
};

__global__ void atrousGradient(const IntegratorGlobals t_globals, int t_stepsize)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x / ASVGF_STRATUM_SIZE) || (current_pix.y >= frame_res.y / ASVGF_STRATUM_SIZE)) return;
	//=========================================================

	int2 grad_pix = current_pix;
	int2 grad_res = frame_res / ASVGF_STRATUM_SIZE;

	int2 pos_in_stratum = make_int2(1);
	int2 sampling_pix = grad_pix * ASVGF_STRATUM_SIZE + pos_in_stratum;
	sampling_pix = clamp(sampling_pix, { 0,0 }, frame_res - 1);

	int sampled_objID = texReadNearest(t_globals.FrameBuffer.objectID_surfobject, sampling_pix).x;
	//void sample/ miss/ sky
	if (sampled_objID < 0) {
		//no filter

		texWrite(make_float4(0, 0, 0, 1),
			t_globals.FrameBuffer.asvgf_dense_gradient_back_surfobject, grad_pix);
		return;
	}

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(t_globals.FrameBuffer.depth_surfobject, sampling_pix, frame_res, ASVGF_STRATUM_SIZE).x,
		dFdy(t_globals.FrameBuffer.depth_surfobject, sampling_pix, frame_res, ASVGF_STRATUM_SIZE).x);

	float4 sampled_gradient = texReadNearest(t_globals.FrameBuffer.asvgf_dense_gradient_front_surfobject, grad_pix);

	float3 sampled_normal = make_float3(texReadNearest(t_globals.FrameBuffer.world_normals_surfobject, sampling_pix));
	float sampled_depth = texReadNearest(t_globals.FrameBuffer.depth_surfobject, sampling_pix).x;

	float4 sum_gradient = sampled_gradient;
	float wsum = 1;

	int radius = 1;//3x3 filter

	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++)
		{
			if (x == 0 && y == 0)continue;
			int2 offset = make_int2(x, y) * t_stepsize;

			int2 grad_tap_pix = grad_pix + offset;
			int2 sampling_tap_pix = sampling_pix + (offset * ASVGF_STRATUM_SIZE);

			grad_tap_pix = clamp(grad_tap_pix, make_int2(0, 0), (grad_res - 1));
			sampling_tap_pix = clamp(sampling_tap_pix, make_int2(0, 0), (frame_res - 1));

			float4 tap_gradient = texReadNearest(t_globals.FrameBuffer.asvgf_dense_gradient_front_surfobject, grad_tap_pix);

			float3 tap_normal = make_float3(texReadNearest(t_globals.FrameBuffer.world_normals_surfobject, sampling_tap_pix));
			float tap_depth = texReadNearest(t_globals.FrameBuffer.depth_surfobject, sampling_tap_pix).x;
			int tap_objID = texReadNearest(t_globals.FrameBuffer.objectID_surfobject, sampling_tap_pix).x;

			float nw = normalWeight(sampled_normal, tap_normal);
			float dw = depthWeight(sampled_depth, tap_depth, dgrad, make_float2(offset * ASVGF_STRATUM_SIZE));

			float w = nw * dw * (sampled_objID == tap_objID);
			float h = box_kernel[(x + radius) + (y + radius) * ((2 * radius) + 1)];

			sum_gradient += tap_gradient * h * w;
			wsum += h * w;
		}
	}
	sum_gradient /= wsum;

	texWrite(sum_gradient, t_globals.FrameBuffer.asvgf_dense_gradient_back_surfobject, grad_pix);
}