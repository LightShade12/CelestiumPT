#include "Integrator.cuh"
#include "LightSampler.cuh"
#include "SceneGeometry.cuh"
#include "DeviceMesh.cuh"
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

__device__ RGBSpectrum temporalAccumulation(const IntegratorGlobals& globals) {
}

__device__ void computeVelocity(const IntegratorGlobals& globals, float2 c_uv, int2 ppixel) {
	//float2 ppixel = { c_uv.x * globals.FrameBuffer.resolution.x ,
	//	c_uv.y * globals.FrameBuffer.resolution.y };

	float4 c_lpos = surf2Dread<float4>(globals.FrameBuffer.local_positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	float4 c_objID = surf2Dread<float4>(globals.FrameBuffer.objectID_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);

	float3 l_pos = make_float3(c_lpos);
	float objID = c_objID.x;

	if (objID < 0) {
		surf2Dwrite(make_float4(0, 0, 0, 1),
			globals.FrameBuffer.velocity_render_surface_object,
			ppixel.x * (int)sizeof(float4), ppixel.y);
		return;
	}

	Mat4 c_VP = globals.SceneDescriptor.active_camera->projectionMatrix * globals.SceneDescriptor.active_camera->viewMatrix;
	Mat4 p_VP = globals.SceneDescriptor.active_camera->prev_projectionMatrix *
		globals.SceneDescriptor.active_camera->prev_viewMatrix;
	Mat4 c_M = globals.SceneDescriptor.device_geometry_aggregate->DeviceMeshesBuffer[(int)objID].modelMatrix;
	Mat4 p_M = globals.SceneDescriptor.device_geometry_aggregate->DeviceMeshesBuffer[(int)objID].prev_modelMatrix;

	float4 c_inpos = c_VP * c_M * make_float4(l_pos, 1);//clipspace
	float4 p_inpos = p_VP * p_M * make_float4(l_pos, 1);//clipspace

	float3 c_ndc = make_float3(c_inpos) / c_inpos.w;
	float3 p_ndc = make_float3(p_inpos) / p_inpos.w;

	float2 vel = make_float2(c_ndc) - make_float2(p_ndc);

	float3 velcol = make_float3(0);
	velcol += (vel.x > 0) ? make_float3(vel.x, 0, 0) : make_float3(0, fabsf(vel.x), fabsf(vel.x));
	velcol += (vel.y > 0) ? make_float3(0, vel.y, 0) : make_float3(fabsf(vel.y), 0, fabsf(vel.y));

	surf2Dwrite(make_float4(velcol, 1),
		globals.FrameBuffer.velocity_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__global__ void renderKernel(IntegratorGlobals globals)
{
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 ppixel = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)thread_pixel_coord_x / (float)frameres.x, (float)thread_pixel_coord_y / (float)frameres.y };

	if ((thread_pixel_coord_x >= frameres.x) || (thread_pixel_coord_y >= frameres.y)) return;

	RGBSpectrum sampled_radiance = IntegratorPipeline::evaluatePixelSample(globals, { (float)thread_pixel_coord_x,(float)thread_pixel_coord_y });

	computeVelocity(globals, screen_uv, ppixel);

	if (globals.IntegratorCFG.accumulate) {
		globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] += make_float3(sampled_radiance);
		sampled_radiance = globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] / (globals.frameidx);
	}
	else if (globals.IntegratorCFG.temporal_accumulation) {
		sampled_radiance = temporalAccumulation(globals);
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

__device__ ShapeIntersection IntegratorPipeline::Intersect(const IntegratorGlobals& globals, const Ray& ray, float tmax)
{
	ShapeIntersection payload;
	payload.hit_distance = tmax;

	payload = globals.SceneDescriptor.device_geometry_aggregate->GAS_structure.intersect(globals, ray, tmax);

	if (payload.triangle_idx == -1) {
		return MissStage(globals, ray, payload);
	}

	return ClosestHitStage(globals, ray, payload);
}

__device__ bool IntegratorPipeline::IntersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax)
{
	return globals.SceneDescriptor.device_geometry_aggregate->GAS_structure.intersectP(globals, ray, tmax);
}

__device__ bool IntegratorPipeline::Unoccluded(const IntegratorGlobals& globals, const ShapeIntersection& p0, float3 p1)
{
	Ray ray = p0.spawnRayTo(p1);
	float tmax = length(p1 - ray.getOrigin()) - 0.011f;
	return !(IntegratorPipeline::IntersectP(globals, ray, tmax));
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
	surf2Dwrite(make_float4(si.l_pos, 1),
		globals.FrameBuffer.local_positions_render_surface_object,
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
	uint32_t obj_id_debug = si.object_idx;
	float3 obj_id_color = make_float3(Samplers::get2D_PCGHash(obj_id_debug), Samplers::get1D_PCGHash(++obj_id_debug));
	surf2Dwrite(make_float4(obj_id_color, 1),
		globals.FrameBuffer.objectID_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ void recordGBufferAny(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si) {
	surf2Dwrite(make_float4(si.GAS_debug, 1),
		globals.FrameBuffer.GAS_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(make_float3(si.object_idx), 1),
		globals.FrameBuffer.objectID_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ void recordGBufferMiss(const IntegratorGlobals& globals, float2 ppixel) {
	surf2Dwrite(make_float4(0, 0, 0.0, 1),
		globals.FrameBuffer.positions_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	surf2Dwrite(make_float4(0, 0, 0.0, 1),
		globals.FrameBuffer.local_positions_render_surface_object,
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
	surf2Dwrite(make_float4(0, 0, 0, 1),
		globals.FrameBuffer.objectID_debug_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ RGBSpectrum IntegratorPipeline::LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel)
{
	Ray ray = in_ray;
	bool DI = true;
	RGBSpectrum throughtput(1.f), light(0.f);
	LightSampler light_sampler(
		globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsBuffer,
		globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsCount);

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

		if (primary_surface || !DI)light += payload.Le(wo) * throughtput;

		//get BSDF
		BSDF bsdf = payload.getBSDF(globals);

		if (DI)
		{
			SampledLight sampled_light = light_sampler.sample(Samplers::get1D_PCGHash(seed));
			//handle empty buffer
			if (sampled_light) {
				LightLiSample ls = sampled_light.light->SampleLi(payload, Samplers::get2D_PCGHash(seed));
				if (ls.pdf > 0) {
					float3 wi = ls.wi;
					RGBSpectrum f = bsdf.f(wo, wi) * AbsDot(wi, payload.w_shading_norm);

					float dist = length(payload.w_pos - ls.pLight);
					float dist_sq = dist * dist;
					float cosTheta_emitter = AbsDot(wi, ls.n);
					float Li_sample_pdf = (sampled_light.p * ls.pdf) * (1 / cosTheta_emitter) * dist_sq;
					if (f && Unoccluded(globals, payload, ls.pLight)) {
						light += throughtput * f * ls.L / Li_sample_pdf;
					}
				}
			}
		}
		if (false) {
			size_t light_idx = Samplers::get1D_PCGHash(seed) * globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsCount;
			Light light_s = globals.SceneDescriptor.device_geometry_aggregate->
				DeviceLightsBuffer[min(light_idx, globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsCount - 1)];
			float light_s_pdf = 1.f / globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsCount;
			Triangle* tri = light_s.m_triangle;
			float2 u2 = Samplers::get2D_PCGHash(seed);
			float3 p = (u2.x * tri->vertex0.position) + (u2.y * tri->vertex1.position) + ((1 - u2.x - u2.y) * tri->vertex2.position);
			p = make_float3(1, 3.5, -1) * 100;
			if (Unoccluded(globals, payload, p)) {
				light += RGBSpectrum(0.2);/// light_s_pdf;
			}
		}

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