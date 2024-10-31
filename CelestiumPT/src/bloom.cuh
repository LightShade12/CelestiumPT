#pragma once
#include "cuda_utility.cuh"

#include "storage.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float4 texReadCODAW(cudaSurfaceObject_t t_tex, int2 t_res, int2 t_pix)
{
	//top left
	int2 a_pix = clamp(t_pix + make_int2(-2, 2), make_int2(0), t_res - 1);
	float4 a = texReadBilinear(t_tex, make_float2(a_pix) + 0.5f, t_res, true);
	//top
	int2 b_pix = clamp(t_pix + make_int2(0, 2), make_int2(0), t_res - 1);
	float4 b = texReadBilinear(t_tex, make_float2(b_pix) + 0.5f, t_res, true);
	//top right
	int2 c_pix = clamp(t_pix + make_int2(2, 2), make_int2(0), t_res - 1);
	float4 c = texReadBilinear(t_tex, make_float2(c_pix) + 0.5f, t_res, true);
	//left
	int2 d_pix = clamp(t_pix + make_int2(-2, 0), make_int2(0), t_res - 1);
	float4 d = texReadBilinear(t_tex, make_float2(d_pix) + 0.5f, t_res, true);
	//center
	float4 e = texReadBilinear(t_tex, make_float2(t_pix) + 0.5f, t_res, true);
	//right
	int2 f_pix = clamp(t_pix + make_int2(2, 0), make_int2(0), t_res - 1);
	float4 f = texReadBilinear(t_tex, make_float2(f_pix) + 0.5f, t_res, true);
	//bottom left
	int2 g_pix = clamp(t_pix + make_int2(-2, -2), make_int2(0), t_res - 1);
	float4 g = texReadBilinear(t_tex, make_float2(g_pix) + 0.5f, t_res, true);
	//bottom
	int2 h_pix = clamp(t_pix + make_int2(0, -2), make_int2(0), t_res - 1);
	float4 h = texReadBilinear(t_tex, make_float2(h_pix) + 0.5f, t_res, true);
	//bottom right
	int2 i_pix = clamp(t_pix + make_int2(2, -2), make_int2(0), t_res - 1);
	float4 i = texReadBilinear(t_tex, make_float2(i_pix) + 0.5f, t_res, true);

	//h 4x4 box
	float4 hb1 = lerp(lerp(a, b, 0.5), lerp(d, e, 0.5), 0.5);
	float4 hb2 = lerp(lerp(b, c, 0.5), lerp(e, f, 0.5), 0.5);
	float4 hb3 = lerp(lerp(d, e, 0.5), lerp(g, h, 0.5), 0.5);
	float4 hb4 = lerp(lerp(e, f, 0.5), lerp(h, i, 0.5), 0.5);

	//----
	//top left
	int2 j_pix = clamp(t_pix + make_int2(-1, 1), make_int2(0), t_res - 1);
	float4 j = texReadBilinear(t_tex, make_float2(j_pix) + 0.5f, t_res, true);
	//top
	int2 k_pix = clamp(t_pix + make_int2(1, 1), make_int2(0), t_res - 1);
	float4 k = texReadBilinear(t_tex, make_float2(k_pix) + 0.5f, t_res, true);
	//top right
	int2 l_pix = clamp(t_pix + make_int2(-1, -1), make_int2(0), t_res - 1);
	float4 l = texReadBilinear(t_tex, make_float2(l_pix) + 0.5f, t_res, true);
	//left
	int2 m_pix = clamp(t_pix + make_int2(1, -1), make_int2(0), t_res - 1);
	float4 m = texReadBilinear(t_tex, make_float2(m_pix) + 0.5f, t_res, true);

	//center h 4x4 box
	float4 hb5 = lerp(lerp(j, k, 0.5), lerp(l, m, 0.5), 0.5);

	float4 out = lerp(hb5, lerp(lerp(hb1, hb2, 0.5), lerp(hb3, hb4, 0.5), 0.5), 0.5);

	return out;
}

__global__ void downSample(const IntegratorGlobals t_globals,
	cudaSurfaceObject_t t_src, int2 t_src_res,
	cudaSurfaceObject_t t_dst, int2 t_dst_res)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= t_dst_res.x) || (current_pix.y >= t_dst_res.y)) return;
	//=========================================================

	float2 scale_ratio = make_float2((float)t_src_res.x / t_dst_res.x, (float)t_src_res.y / t_dst_res.y);

	//src-res
	int2 tap_pix = make_int2(current_pix.x * scale_ratio.x, current_pix.y * scale_ratio.y);
	tap_pix = clamp(tap_pix, make_int2(0), t_src_res - 1);

	//min filter
	//float4 color = texReadBilinear(t_src, make_float2(tap_pix) + 0.5f, t_src_res, false);
	float4 color = texReadCODAW(t_src, (t_src_res), tap_pix);

	texWrite(color, t_dst, current_pix);
}

__global__ void upSampleAdd(const IntegratorGlobals t_globals,
	cudaSurfaceObject_t t_src, int2 t_src_res,
	cudaSurfaceObject_t t_dst, int2 t_dst_res)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= t_dst_res.x) || (current_pix.y >= t_dst_res.y)) return;
	//=========================================================

	float2 uv = make_float2(
		(float)current_pix.x / (float)t_dst_res.x,
		(float)current_pix.y / (float)t_dst_res.y
	);

	float2 src_pixf = make_float2(uv.x * t_src_res.x, uv.y * t_src_res.y);

	src_pixf = clamp(src_pixf, make_float2(0), make_float2(t_src_res - 1));

	float4 final_col = make_float4(0.0f);
	//float4 final_col = texReadBilinear(t_src, src_pixf, t_src_res, true);

	// Define the 3x3 tent filter weights/gaussian blur
	constexpr float filter[3][3] = {
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
		{2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
	};

	//TODO: make separable gaussian blur?
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			float2 tap_pix = src_pixf + make_float2(x, y);
			tap_pix = clamp(tap_pix, make_float2(0), make_float2(t_src_res) - 1.f);

			float4 tap_col = texReadBilinear(t_src, tap_pix, t_src_res, true);

			final_col += filter[y + 1][x + 1] * tap_col;
		}
	}

	//combine prev mip
	float4 col = texReadNearest(t_dst, current_pix);

	//final_col += col;
	final_col = lerp(final_col, col, 0.85);

	texWrite(final_col, t_dst, current_pix);
}

__global__ void upSample(const IntegratorGlobals t_globals,
	cudaSurfaceObject_t t_src, int2 t_src_res,
	cudaSurfaceObject_t t_dst, int2 t_dst_res)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= t_dst_res.x) || (current_pix.y >= t_dst_res.y)) return;
	//=========================================================

	float2 uv = make_float2(
		(float)current_pix.x / (float)t_dst_res.x,
		(float)current_pix.y / (float)t_dst_res.y
	);

	float2 src_pixf = make_float2(uv.x * t_src_res.x, uv.y * t_src_res.y);

	src_pixf = clamp(src_pixf, make_float2(0), make_float2(t_src_res - 1));

	float4 final_col = make_float4(0.0f);
	//float4 final_col = texReadBilinear(t_src, src_pixf, t_src_res, true);

	// Define the 3x3 tent filter weights/gaussian blur
	constexpr float filter[3][3] = {
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
		{2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
		{1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
	};

	//TODO: make separable gaussian blur?
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			float2 tap_pix = src_pixf + make_float2(x, y);
			tap_pix = clamp(tap_pix, make_float2(0), make_float2(t_src_res) - 1.f);

			float4 tap_col = texReadBilinear(t_src, tap_pix, t_src_res, true);

			final_col += filter[y + 1][x + 1] * tap_col;
		}
	}

	texWrite(final_col, t_dst, current_pix);
}