#include "Film.cuh"
#include "ShapeIntersection.cuh"
#include "SceneGeometry.cuh"
#include "DeviceMaterial.cuh"
#include "Storage.cuh"
#include "Samplers.cuh"
#include "maths/matrix.cuh"

#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <float.h>

// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

// Mean error^2: 3.6705141e-06
__device__ float3 agxDefaultContrastApprox(float3 x) {
	float3 x2 = x * x;
	float3 x4 = x2 * x2;

	return +15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}

__device__ RGBSpectrum agx_fitted(RGBSpectrum	col) {
	float3 val = make_float3(col);
	const Mat3 agx_mat = Mat3(
		0.842479062253094, 0.0423282422610123, 0.0423756549057051,
		0.0784335999999992, 0.878468636469772, 0.0784336,
		0.0792237451477643, 0.0791661274605434, 0.879142973793104);

	const float min_ev = -12.47393f;
	const float max_ev = 4.026069f;

	// Input transform (inset)
	val = agx_mat * val;

	// Log2 space encoding
	val = clamp(log2f(val), min_ev, max_ev);
	val = (val - min_ev) / (max_ev - min_ev);

	// Apply sigmoid function approximation
	val = agxDefaultContrastApprox(val);

	return RGBSpectrum(val);
}

__device__ RGBSpectrum agx_fitted_Eotf(RGBSpectrum col) {
	float3 val = make_float3(col);
	const Mat3 agx_mat_inv = Mat3(
		1.19687900512017, -0.0528968517574562, -0.0529716355144438,
		-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
		-0.0990297440797205, -0.0989611768448433, 1.15107367264116);

	// Inverse input transform (outset)
	val = agx_mat_inv * val;

	// sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
	// NOTE: We're linearizing the output here. Comment/adjust when
	// *not* using a sRGB render target
	val = powf(val, make_float3(2.2));

	return RGBSpectrum(val);
}

__device__ float3 agxLook(float3 val) {
	const float3 lw = make_float3(0.2126, 0.7152, 0.0722);
	float luma = dot(val, lw);

	// Default
	float3 offset = make_float3(0.0);
	float3 slope = make_float3(1.0);
	float3 power = make_float3(1.0);
	float sat = 1.0;

#if AGX_LOOK == 1
	// Golden
	slope = make_float3(1.0, 0.9, 0.5);
	power = make_float3(0.8);
	sat = 0.8;
#elif AGX_LOOK == 2
	// Punchy
	slope = make_float3(1.0);
	power = make_float3(1.35, 1.35, 1.35);
	sat = 1.4;
#endif

	// ASC CDL
	val = powf(val * slope + offset, power);
	return luma + sat * (val - luma);
}

//algebraic solution
__device__ float agx_curve(float x) {
	if (x >= (20.0 / 33.0))
		return 0.5 + (2.0 * (-(20.0 / 33.0) + x)) / pow(1.0 + 69.86278913545539 * pow(-(20.0 / 33.0) + x, (13.0 / 4.0)), (4.0 / 13.0));
	else
		return 0.5 + (2.0 * (-(20.0 / 33.0) + x)) / pow(1.0 - 59.507875 * pow(-(20.0 / 33.0) + x, (3.0 / 1.0)), (1.0 / 3.0));
}

__device__ float3 agx_curve3(float3 v) {
	return make_float3(agx_curve(v.x), agx_curve(v.y), agx_curve(v.z));
}

__device__ RGBSpectrum agx_tonemapping(RGBSpectrum /*Linear BT.709*/col) {
	float3 val = make_float3(col);
	const Mat3 agx_mat = Mat3(
		0.842479062253094, 0.0423282422610123, 0.0423756549057051,
		0.0784335999999992, 0.878468636469772, 0.0784336,
		0.0792237451477643, 0.0791661274605434, 0.879142973793104);
	const Mat3 agx_mat_inv = Mat3(
		1.19687900512017, -0.0528968517574562, -0.0529716355144438,
		-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
		-0.0990297440797205, -0.0989611768448433, 1.15107367264116);
	const float min_ev = -12.47393;
	const float max_ev = 4.026069;

	// Input transform (inset)
	val = agx_mat * val;

	// Log2 space encoding
	val = clamp(log2f(val) * (1.0 / (max_ev - min_ev)) - (min_ev / (max_ev - min_ev)), make_float3(0.0), make_float3(1.0));

	// Apply sigmoid function
	float3 res = agx_curve3(val);

	// Inverse input transform (outset)
	res = agx_mat_inv * res;

	return /*Linear BT.709*/RGBSpectrum(res);
}

__device__ RGBSpectrum uncharted2_tonemap_partial(RGBSpectrum x)
{
	constexpr float A = 0.15f;
	constexpr float B = 0.50f;
	constexpr float C = 0.10f;
	constexpr float D = 0.20f;
	constexpr float E = 0.02f;
	constexpr float F = 0.30f;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

__device__ RGBSpectrum uncharted2_filmic(RGBSpectrum v, float exposure)
{
	float exposure_bias = exposure;
	RGBSpectrum curr = uncharted2_tonemap_partial(v * exposure_bias);

	RGBSpectrum W(11.2f);
	RGBSpectrum white_scale = RGBSpectrum(1.0f) / uncharted2_tonemap_partial(W);
	return curr * white_scale;
}

__device__ RGBSpectrum toneMapping(RGBSpectrum HDR_color, float exposure)
{
	RGBSpectrum LDR_color = uncharted2_filmic(HDR_color, exposure);
	return LDR_color;
}

__device__ RGBSpectrum gammaCorrection(const RGBSpectrum linear_color)
{
	RGBSpectrum gamma_space_color = { sqrtf(linear_color.r),sqrtf(linear_color.g) ,sqrtf(linear_color.b) };
	return gamma_space_color;
}

__device__ void recordGBufferHit(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si)
{
	const Triangle& triangle = globals.SceneDescriptor.device_geometry_aggregate->DeviceTrianglesBuffer[si.triangle_idx];
	DeviceMaterial& material = globals.SceneDescriptor.device_geometry_aggregate->DeviceMaterialBuffer[triangle.mat_idx];

	surf2Dwrite((material.emission_color_factor) ?
		//make_float4(material.emission_color_factor * material.emission_strength, 1)
		make_float4(1)
		: make_float4(make_float3(material.albedo_color_factor) / PI, 1),
		globals.FrameBuffer.albedo_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.w_pos, 1),
		globals.FrameBuffer.positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(make_float3(si.hit_distance), 1),
		globals.FrameBuffer.depth_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.l_pos, 1),
		globals.FrameBuffer.local_positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.w_shading_norm, 1),
		globals.FrameBuffer.world_normals_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.l_shading_norm, 1),
		globals.FrameBuffer.local_normals_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.bary, 1),
		globals.FrameBuffer.bary_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.uv.x, si.uv.y, 0, 1),
		globals.FrameBuffer.UV_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	uint32_t obj_id_debug = si.object_idx;
	float3 obj_id_color = make_float3(Samplers::get2D_PCGHash(obj_id_debug), Samplers::get1D_PCGHash(++obj_id_debug));
	surf2Dwrite(make_float4(obj_id_color, 1),
		globals.FrameBuffer.objectID_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ void recordGBufferAny(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si)
{
	//float2 uv = ppixel / make_float2(globals.FrameBuffer.resolution);
	//float3 dbg_uv_col = make_float3(uv);

	surf2Dwrite(make_float4(si.GAS_debug, 1),
		globals.FrameBuffer.GAS_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(make_float3(si.object_idx), 1),
		globals.FrameBuffer.objectID_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ void recordGBufferMiss(const IntegratorGlobals& globals, float2 ppixel)
{
	surf2Dwrite(make_float4(0, 0, 0.0, 1),
		globals.FrameBuffer.positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(1.f),
		globals.FrameBuffer.albedo_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(make_float3(FLT_MAX), 1),
		globals.FrameBuffer.depth_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0.0, 1),
		globals.FrameBuffer.local_positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.world_normals_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.local_normals_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.bary_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.UV_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.objectID_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}