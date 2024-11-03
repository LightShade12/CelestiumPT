#include "film.cuh"
#include "shape_intersection.cuh"
#include "scene_geometry.cuh"
#include "device_material.cuh"
#include "device_texture.cuh"
#include "storage.cuh"
#include "samplers.cuh"
#include "cuda_utility.cuh"
#include "maths/matrix_maths.cuh"

#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <float.h>

__device__ float3 hueToRGB(float hue) {
	// Clamp hue to the [0, 1] range to avoid issues
	hue = fmodf(hue, 1.0f);
	if (hue < 0.0f) hue += 1.0f;

	float s = 1.0f;  // Full saturation
	float v = 1.0f;  // Full brightness

	// Offset the hue to start from blue and move counterclockwise through cyan to red
	// Adding 2/3 (240°) makes 0 correspond to blue.
	float h = (hue + 2.0f / 3.0f) * 6.0f;  // Hue is now in [0, 6] range
	int i = int(floorf(h));  // Which sector of the hue circle are we in
	float f = h - i;         // Fractional part of h

	float p = v * (1.0f - s);
	float q = v * (1.0f - s * f);
	float t = v * (1.0f - s * (1.0f - f));

	float3 color;
	switch (i % 6) {  // Ensure 'i' is within [0, 5] range
	case 0: color = make_float3(v, t, p); break;   // Red -> Yellow
	case 1: color = make_float3(q, v, p); break;   // Yellow -> Green
	case 2: color = make_float3(p, v, t); break;   // Green -> Cyan
	case 3: color = make_float3(p, q, v); break;   // Cyan -> Blue
	case 4: color = make_float3(t, p, v); break;   // Blue -> Magenta
	case 5: color = make_float3(v, p, q); break;   // Magenta -> Red
	default: color = make_float3(1.0f, 0.0f, 0.0f);  // Red in case of unexpected values
	}

	return color;
}

namespace AgxMinimal
{
	// 0: Default, 1: Golden, 2: Punchy
#define AGX_LOOK 0

//Fifth order
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

	__device__ float3 agxLook(float3 val) 
	{
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
}

//algebraic solution
namespace AgxAlgebraic {
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

__device__ void recordGBufferHit(const IntegratorGlobals& globals, int2 ppixel, const ShapeIntersection& si)
{
	const Triangle& triangle = globals.SceneDescriptor.DeviceGeometryAggregate->DeviceTrianglesBuffer[si.triangle_idx];
	DeviceMaterial& material = globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMaterialBuffer[triangle.mat_idx];

	float3 albedo = make_float3(material.albedo_color_factor);
	if (material.albedo_color_texture_id >= 0)
	{
		const auto& tex = globals.SceneDescriptor.DeviceGeometryAggregate->DeviceTexturesBuffer[material.albedo_color_texture_id];
		albedo = (tex.sampleNearest(si.uv, false));
	}

	texWrite((material.emission_color_factor * material.emission_strength) ?
		make_float4(1)
		: make_float4(albedo / PI, 1),
		globals.FrameBuffer.albedo_surfobject, ppixel);

	texWrite(make_float4(si.w_pos, 1),
		globals.FrameBuffer.world_positions_surfobject, ppixel);

	texWrite(make_float4(make_float3(si.hit_distance), 1),
		globals.FrameBuffer.depth_surfobject, ppixel);

	texWrite(make_float4(si.l_pos, 1),
		globals.FrameBuffer.local_positions_surfobject, ppixel);

	texWrite(make_float4(si.w_shading_norm, 1),
		globals.FrameBuffer.world_normals_surfobject, ppixel);

	texWrite(make_float4(si.l_shading_norm, 1),
		globals.FrameBuffer.local_normals_surfobject, ppixel);

	texWrite(make_float4(si.bary, 1),
		globals.FrameBuffer.bary_surfobject, ppixel);

	texWrite(make_float4(si.uv.x, si.uv.y, 0, 1),
		globals.FrameBuffer.UVs_surfobject, ppixel);

	uint32_t obj_id_debug = si.object_idx;
	float3 obj_id_color = make_float3(Samplers::get2D_PCGHash(obj_id_debug), Samplers::get1D_PCGHash(++obj_id_debug));
	texWrite(make_float4(obj_id_color, 1),
		globals.FrameBuffer.debugview_objectID_surfobject, ppixel);
}

__device__ void recordGBufferAny(const IntegratorGlobals& globals, int2 ppixel, const ShapeIntersection& si)
{
	//Gbuffer
	texWrite(make_float4(make_float3(si.object_idx), 1),
		globals.FrameBuffer.objectID_surfobject, ppixel);
	texWrite(make_float4(make_float3(si.triangle_idx), 1),
		globals.FrameBuffer.triangleID_surfobject, ppixel);

	//DebugViews===================================

	//float2 uv = ppixel / make_float2(globals.FrameBuffer.resolution);
	//float3 dbg_uv_col = make_float3(uv);
	float3 heatmap;
	float3 bboxheatmap;
	int hit_threshold = 90;
	int bbox_threshold = 20;
	if (si.hit_count == 0) {
		heatmap = { 0,0,0 };
		bboxheatmap = { 0,0,0 };
	}
	else {
		//heatmap = lerp(make_float3(0, 0, 1), make_float3(1, 0, 0),
		//	fminf(1.f, si.hit_count / (float)threshold));
		heatmap = hueToRGB(1.f - (si.hit_count / (float)hit_threshold));
		bboxheatmap = hueToRGB(1.f - (si.bbox_hit_count / (float)bbox_threshold));
	}
	texWrite(make_float4(bboxheatmap, 1),
		globals.FrameBuffer.debugview_bbox_test_heatmap_surfobject, ppixel);
	texWrite(make_float4(heatmap, 1),
		globals.FrameBuffer.debugview_tri_test_heatmap_surfobject, ppixel);
	texWrite(make_float4(si.GAS_debug, 1),
		globals.FrameBuffer.debugview_GAS_overlap_surfobject, ppixel);
}

__device__ void recordGBufferMiss(const IntegratorGlobals& globals, int2 ppixel)
{
	texWrite(make_float4(1.f),
		globals.FrameBuffer.albedo_surfobject, ppixel);

	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.world_positions_surfobject, ppixel);
	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.local_positions_surfobject, ppixel);
	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.world_normals_surfobject, ppixel);
	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.local_normals_surfobject, ppixel);

	texWrite(make_float4(make_float3(-1), 1),
		globals.FrameBuffer.depth_surfobject, ppixel);

	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.bary_surfobject, ppixel);
	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.UVs_surfobject, ppixel);

	//DebugView-----------------------------------
	texWrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.debugview_objectID_surfobject, ppixel);
}