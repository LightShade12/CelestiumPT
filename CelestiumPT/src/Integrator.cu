#include "Integrator.cuh"
#include "SceneGeometry.cuh"
#include "DeviceCamera.cuh"
#include "Storage.cuh"
#include "RayStages.cuh"
#include "Ray.cuh"
#include "ShapeIntersection.cuh"
#include "Spectrum.cuh"
#include "BSDF.cuh"
#include "acceleration_structure/GAS.cuh"
#include "Samplers.cuh"

#include "maths/maths_linear_algebra.cuh"
#include "maths/constants.cuh"

#include <device_launch_parameters.h>
#define __CUDACC__
#include <surface_indirect_functions.h>
#include <float.h>

__device__ static RGBSpectrum uncharted2_tonemap_partial(RGBSpectrum x)
{
	constexpr float A = 0.15f;
	constexpr float B = 0.50f;
	constexpr float C = 0.10f;
	constexpr float D = 0.20f;
	constexpr float E = 0.02f;
	constexpr float F = 0.30f;
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

__device__ static RGBSpectrum uncharted2_filmic(RGBSpectrum v, float exposure)
{
	float exposure_bias = exposure;
	RGBSpectrum curr = uncharted2_tonemap_partial(v * exposure_bias);

	RGBSpectrum W = make_float3(11.2f);
	RGBSpectrum white_scale = RGBSpectrum(1.0f) / uncharted2_tonemap_partial(W);
	return curr * white_scale;
}

__device__ static RGBSpectrum toneMapping(RGBSpectrum HDR_color, float exposure = 2.f) {
	RGBSpectrum LDR_color = uncharted2_filmic(HDR_color, exposure);
	return LDR_color;
}

__device__ static RGBSpectrum gammaCorrection(const RGBSpectrum linear_color) {
	RGBSpectrum gamma_space_color = { sqrtf(linear_color.r),sqrtf(linear_color.g) ,sqrtf(linear_color.b) };
	return gamma_space_color;
}

__host__ void IntegratorPipeline::invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims)
{
	renderKernel << < block_grid_dims, thread_block_dims >> > (globals);
};

__global__ void renderKernel(IntegratorGlobals globals)
{
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 frameres = globals.FrameBuffer.resolution;

	if ((thread_pixel_coord_x >= frameres.x) || (thread_pixel_coord_y >= frameres.y)) return;

	RGBSpectrum sampled_radiance = IntegratorPipeline::evaluatePixelSample(globals, { (float)thread_pixel_coord_x,(float)thread_pixel_coord_y });

	if (globals.IntegratorCFG.accumulate) {
		globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] += make_float3(sampled_radiance);
		sampled_radiance = globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] / (globals.frameidx);
	}

	float4 fragcolor = { sampled_radiance.r,sampled_radiance.g,sampled_radiance.b, 1 };

	//EOTF
	fragcolor = make_float4(gammaCorrection(make_float3(fragcolor)), 1);
	fragcolor = make_float4(toneMapping(make_float3(fragcolor), 8), 1);
	//fragcolor = make_float4(sqrtf(sampled_radiance.x), sqrtf(sampled_radiance.y), sqrtf(sampled_radiance.z), 1);

	surf2Dwrite(fragcolor, globals.FrameBuffer.composite_render_surface_object, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}

__device__ RGBSpectrum IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel)
{
	uint32_t seed = ppixel.x + ppixel.y * globals.FrameBuffer.resolution.x;
	seed *= globals.frameidx;

	int2 frameres = globals.FrameBuffer.resolution;

	float2 screen_uv = { (ppixel.x / frameres.x),(ppixel.y / frameres.y) };
	screen_uv = screen_uv * 2 - 1;//-1->1

	Ray primary_ray = globals.SceneDescriptor.active_camera->generateRay(frameres.x, frameres.y, screen_uv);

	RGBSpectrum L = IntegratorPipeline::Li(globals, primary_ray, seed, ppixel);

	return L;
}

__device__ ShapeIntersection IntegratorPipeline::Intersect(const IntegratorGlobals& globals, const Ray& ray)
{
	ShapeIntersection payload;
	payload.hit_distance = FLT_MAX;

	payload = globals.SceneDescriptor.device_geometry_aggregate->GAS_structure.intersect(globals, ray);

	if (payload.triangle_idx == -1) {
		return MissStage(globals, ray, payload);
	}

	return ClosestHitStage(globals, ray, payload);
}

__device__ bool IntegratorPipeline::IntersectP(const IntegratorGlobals& globals, const Ray& ray)
{
	return false;
}

__device__ bool IntegratorPipeline::Unoccluded(const IntegratorGlobals& globals, const Ray& ray)
{
	return !(IntegratorPipeline::IntersectP(globals, ray));
}

__device__ RGBSpectrum IntegratorPipeline::Li(const IntegratorGlobals& globals, const Ray& ray, uint32_t seed, float2 ppixel)
{
	//return make_float3(1, 0, 1);
	return IntegratorPipeline::LiRandomWalk(globals, ray, seed, ppixel);
}

__device__ RGBSpectrum SkyShading(const Ray& ray) {
	//return make_float3(0);
	float3 unit_direction = normalize(ray.getDirection());
	float a = 0.5f * (unit_direction.y + 1.0);
	//return make_float3(0.2f, 0.3f, 0.4f);
	return (1.0f - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.2, 0.4, 1.0);
};

__device__ void recordGBufferHit(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si) {
	surf2Dwrite(make_float4(si.w_pos, 1),
		globals.FrameBuffer.positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.w_shading_norm, 1),
		globals.FrameBuffer.normals_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.bary, 1),
		globals.FrameBuffer.bary_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(si.uv.x, si.uv.y, 0, 1),
		globals.FrameBuffer.UV_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ void recordGBufferAny(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si) {
	surf2Dwrite(make_float4(si.GAS_debug, 1),
		globals.FrameBuffer.GAS_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ void recordGBufferMiss(const IntegratorGlobals& globals, float2 ppixel) {
	surf2Dwrite(make_float4(0, 0, 0.5, 1),
		globals.FrameBuffer.positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.normals_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.bary_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.UV_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ RGBSpectrum IntegratorPipeline::LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel)
{
	Ray ray = in_ray;

	RGBSpectrum throughtput(1.f), light(0.f);

	ShapeIntersection payload{};

	for (int bounce_depth = 0; bounce_depth <= globals.IntegratorCFG.bounces; bounce_depth++) {
		seed += bounce_depth;
		payload = IntegratorPipeline::Intersect(globals, ray);

		bool primary_surface = (bounce_depth == 0);

		if (primary_surface) recordGBufferAny(globals, ppixel, payload);

		//miss--
		if (payload.hit_distance < 0)//TODO: standardize invalid/miss payload definition
		{
			if (primary_surface) recordGBufferMiss(globals, ppixel);

			light += SkyShading(ray) * throughtput;
			break;
		}

		//hit--
		if (primary_surface) recordGBufferHit(globals, ppixel, payload);

		float3 wo = -ray.getDirection();
		light += payload.Le(wo) * throughtput;

		//get BSDF
		BSDF bsdf = payload.getBSDF(globals);
		BSDFSample bs = bsdf.sampleBSDF(wo, Samplers::get2D_PCGHash(seed));

		float3 wi = bs.wi;

		RGBSpectrum fcos = bs.f * AbsDot(wi, payload.w_shading_norm);
		if (!fcos)break;

		float pdf = bs.pdf;

		throughtput *= fcos / pdf;

		ray = payload.spawnRay(wi);
	}

	return light;
};