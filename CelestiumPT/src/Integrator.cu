#include "integrator.cuh"
#include "film.cuh"
#include "light_sampler.cuh"
#include "scene_geometry.cuh"
#include "device_mesh.cuh"
#include "device_camera.cuh"
#include "storage.cuh"
#include "ray_stages.cuh"
#include "ray.cuh"
#include "shape_intersection.cuh"
#include "spectrum.cuh"
#include "bsdf.cuh"
#include "acceleration_structure/GAS.cuh"
#include "samplers.cuh"
#include "cuda_utility.cuh"

#include "maths/linear_algebra.cuh"
#include "maths/sampling.cuh"
#include "maths/constants.cuh"

#include <device_launch_parameters.h>
#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <float.h>

__device__ RGBSpectrum IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, int2 ppixel)
{
	int2 frameres = globals.FrameBuffer.resolution;

	uint32_t seed = ppixel.x + ppixel.y * frameres.x;
	seed *= globals.FrameIndex;

	float2 screen_uv = { (ppixel.x / frameres.x),(ppixel.y / frameres.y) };
	screen_uv = screen_uv * 2 - 1;//-1->1

	Ray primary_ray = globals.SceneDescriptor.ActiveCamera->generateRay(frameres.x, frameres.y, screen_uv);

	RGBSpectrum L = IntegratorPipeline::LiPathIntegrator(globals, primary_ray, seed, ppixel);

	return L;
}

__device__ ShapeIntersection IntegratorPipeline::Intersect(const IntegratorGlobals& globals, const Ray& ray, float tmax)
{
	ShapeIntersection payload;
	payload.hit_distance = tmax;

	payload = globals.SceneDescriptor.DeviceGeometryAggregate->GAS_structure.intersect(globals, ray, tmax);

	if (payload.triangle_idx == -1) {
		return MissStage(globals, ray, payload);
	}

	return ClosestHitStage(globals, ray, payload);
}

__device__ bool IntegratorPipeline::IntersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax)
{
	return globals.SceneDescriptor.DeviceGeometryAggregate->GAS_structure.intersectP(globals, ray, tmax);
}

__device__ bool IntegratorPipeline::Unoccluded(const IntegratorGlobals& globals, const ShapeIntersection& p0, float3 p1)
{
	Ray ray = p0.spawnRayTo(p1);
	float tmax = length(p1 - ray.getOrigin()) - 0.011f;
	return !(IntegratorPipeline::IntersectP(globals, ray, tmax));
}

__device__ RGBSpectrum IntegratorPipeline::LiPathIntegrator(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, int2 ppixel)
{
	Ray ray = in_ray;
	RGBSpectrum throughtput(1.f), light(0.f);
	LightSampler light_sampler(
		globals.SceneDescriptor.DeviceGeometryAggregate->DeviceLightsBuffer,
		globals.SceneDescriptor.DeviceGeometryAggregate->DeviceLightsCount);
	float p_b = 1;
	LightSampleContext prev_ctx{};
	ShapeIntersection payload{};
	float eta_scale = 1;//TODO: look up russian roulette
	//float3 sunpos = make_float3(sinf(t_globals.FrameIndex * 0.01f), 1, cosf(t_globals.FrameIndex * 0.01f)) * 100;
	float3 sunpos = make_float3(0.266, 0.629, 0.257) * 100;
	RGBSpectrum suncol(1.000, 0.877, 0.822);
	//suncol *= RGBSpectrum(1, 0.7, 0.4);

	for (int bounce_depth = 0; bounce_depth <= globals.IntegratorCFG.max_bounces; bounce_depth++) {
		seed += bounce_depth;

		payload = IntegratorPipeline::Intersect(globals, ray);

		bool primary_surface = (bounce_depth == 0);

		if (primary_surface) recordGBufferAny(globals, ppixel, payload);

		//miss--
		if (payload.hit_distance < 0)//TODO: standardize invalid/miss payload definition
		{
			if (primary_surface) recordGBufferMiss(globals, ppixel);

			light += globals.SceneDescriptor.DeviceGeometryAggregate->SkyLight.Le(ray) * RGBSpectrum(0.8, 1, 1.5) * throughtput;
			break;
		}

		//hit--
		if (primary_surface) recordGBufferHit(globals, ppixel, payload);

		float3 wo = -ray.getDirection();

		RGBSpectrum Le = payload.Le(wo);

		if (Le) {
			if (primary_surface)
				light += Le * throughtput;
			else {
				const Light* arealight = payload.arealight;
				float light_pdf = light_sampler.PMF(arealight) * arealight->PDF_Li(prev_ctx, ray.getDirection());
				//---
				float dist = length(prev_ctx.pos - payload.w_pos);
				float cosTheta_emitter = AbsDot(-ray.getDirection(), payload.w_geo_norm);
				light_pdf = light_pdf * (1.f / cosTheta_emitter) * Sqr(dist);
				//---
				float w_l = powerHeuristic(1, p_b, 1, light_pdf);
				light += Le * throughtput * w_l;
			}
		}

		BSDF bsdf = payload.getBSDF(globals);

		RGBSpectrum Ld = SampleLd(globals, ray, payload, bsdf,
			light_sampler, seed, primary_surface);
		light += Ld * throughtput;

		if (false) {
			bool sunhit = !IntersectP(globals, Ray(payload.w_pos + payload.w_geo_norm * 0.001f,
				sunpos + make_float3(Samplers::get2D_PCGHash(seed), Samplers::get1D_PCGHash(seed)) * 5.f),
				100);
			if (sunhit) {
				RGBSpectrum f_c = suncol * bsdf.f(wo, normalize(sunpos), primary_surface)
					* dot(payload.w_shading_norm, normalize(sunpos)) * 20.f;
				light += f_c * throughtput;
			}
		}

		BSDFSample bs = bsdf.sampleBSDF(wo, Samplers::get2D_PCGHash(seed), primary_surface);
		float3 wi = bs.wi;
		float pdf = bs.pdf;
		if (primary_surface)bs.f = RGBSpectrum(1);
		RGBSpectrum fcos = bs.f * AbsDot(wi, payload.w_shading_norm);
		if (!fcos)break;
		throughtput *= fcos / pdf;

		p_b = bs.pdf;
		prev_ctx = LightSampleContext(payload);

		ray = payload.spawnRay(wi);

		RGBSpectrum RR_beta = throughtput * eta_scale;
		if (RR_beta.maxComponentValue() < 1 && bounce_depth > 1) {
			float q = fmaxf(0.f, 1.f - RR_beta.maxComponentValue());
			if (Samplers::get1D_PCGHash(seed) < q)
				break;
			throughtput /= 1 - q;
		}
	}

	return clampOutput(light);
}

__device__ ShapeIntersection initializePayloadFromGBuffer(const IntegratorGlobals& t_globals, int2 t_current_pix)
{
	ShapeIntersection out_payload;
	out_payload.w_pos = make_float3(texReadNearest(t_globals.FrameBuffer.world_positions_surfobject,
		(t_current_pix)));
	out_payload.l_pos = make_float3(texReadNearest(t_globals.FrameBuffer.local_positions_surfobject,
		(t_current_pix)));

	out_payload.w_shading_norm = make_float3(texReadNearest(t_globals.FrameBuffer.world_normals_surfobject,
		(t_current_pix)));
	out_payload.l_shading_norm = make_float3(texReadNearest(t_globals.FrameBuffer.local_normals_surfobject,
		(t_current_pix)));

	//----------
	out_payload.triangle_idx = texReadNearest(t_globals.FrameBuffer.triangleID_surfobject,
		(t_current_pix)).x;
	out_payload.object_idx = texReadNearest(t_globals.FrameBuffer.objectID_surfobject,
		(t_current_pix)).x;

	//----------
	out_payload.hit_distance = texReadNearest(t_globals.FrameBuffer.depth_surfobject,
		(t_current_pix)).x;

	//----------
	out_payload.bary = make_float3(texReadNearest(t_globals.FrameBuffer.bary_surfobject,
		(t_current_pix)));

	out_payload.uv = make_float2(make_float3(texReadNearest(t_globals.FrameBuffer.UVs_surfobject,
		(t_current_pix))));

	//----------
	out_payload.arealight = nullptr;

	if (out_payload.triangle_idx >= 0) {
		const Triangle& triangle = t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceTrianglesBuffer[out_payload.triangle_idx];

		if (triangle.LightIdx >= 0) {
			out_payload.arealight =
				&(t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceLightsBuffer[triangle.LightIdx]);
		}

		const DeviceMesh& mesh = t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMeshesBuffer[out_payload.object_idx];

		//---------
		out_payload.m_invModelMatrix = mesh.inverseModelMatrix;

		//construct geometric normal
		out_payload.w_geo_norm = normalize(make_float3(
			out_payload.m_invModelMatrix.inverse() * make_float4(triangle.face_normal, 0)));

		//shading may have been flipped from init geo; if diff then was flipped i.e backface
		if (dot(out_payload.w_geo_norm, out_payload.w_shading_norm) < 0) {
			out_payload.front_face = false;
			out_payload.w_geo_norm = -1.f * out_payload.w_geo_norm;
		}

		//---------
	}

	return out_payload;
};

__device__ RGBSpectrum IntegratorPipeline::deferredEvaluatePixelSample(const IntegratorGlobals& t_globals, int2 t_current_pix, uint32_t t_seed)
{
	float3 cpos = make_float3(t_globals.SceneDescriptor.ActiveCamera->invViewMatrix[3]);

	ShapeIntersection payload = initializePayloadFromGBuffer(t_globals, t_current_pix);

	Ray ray = Ray(cpos, normalize(payload.w_pos - cpos));

	RGBSpectrum throughtput(1.f), light(0.f);

	LightSampler light_sampler(
		t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceLightsBuffer,
		t_globals.SceneDescriptor.DeviceGeometryAggregate->DeviceLightsCount);

	float p_b = 1;
	LightSampleContext prev_ctx{};
	float eta_scale = 1;

	float3 sunpos = make_float3(sinf(t_globals.FrameIndex * 0.01f), 1, cosf(t_globals.FrameIndex * 0.01f)) * 100;
	//float3 sunpos = make_float3(0.266, 0.629, 0.257) * 100;
	RGBSpectrum suncol(1.000, 0.877, 0.822);

	for (int bounce_depth = 0; bounce_depth <= t_globals.IntegratorCFG.max_bounces; bounce_depth++)
	{
		t_seed += bounce_depth;

		bool primary_surface = (bounce_depth == 0);

		if (!primary_surface)payload = IntegratorPipeline::Intersect(t_globals, ray);

		//miss--
		if (payload.hit_distance < 0)//TODO: standardize invalid/miss payload definition
		{
			if (t_globals.IntegratorCFG.skylight_enabled)
			{
				light += t_globals.SceneDescriptor.DeviceGeometryAggregate->SkyLight.Le(ray)
					* throughtput * t_globals.IntegratorCFG.skylight_intensity;
			}
			break;
		}

		//hit--

		float3 wo = -ray.getDirection();

		RGBSpectrum Le = payload.Le(wo);

		if (Le) {
			if (primary_surface)
				light += Le * throughtput;
			else {
				const Light* arealight = payload.arealight;
				float light_pdf = light_sampler.PMF(arealight) * arealight->PDF_Li(prev_ctx, ray.getDirection());
				//---
				float dist = length(prev_ctx.pos - payload.w_pos);
				float cosTheta_emitter = AbsDot(-ray.getDirection(), payload.w_geo_norm);
				light_pdf = light_pdf * (1.f / cosTheta_emitter) * Sqr(dist);
				//---
				float w_l = powerHeuristic(1, p_b, 1, light_pdf);
				light += Le * throughtput * w_l;
			}
		}

		BSDF bsdf = payload.getBSDF(t_globals);

		RGBSpectrum Ld = SampleLd(t_globals, ray, payload, bsdf,
			light_sampler, t_seed, primary_surface);
		light += Ld * throughtput;

		//sun sample
		if (t_globals.IntegratorCFG.sunlight_enabled)
		{
			bool sunhit = !IntersectP(t_globals, Ray(payload.w_pos + payload.w_geo_norm * 0.001f,
				sunpos + make_float3(Samplers::get2D_PCGHash(t_seed), Samplers::get1D_PCGHash(t_seed)) * 5.f),
				100);
			if (sunhit) {
				RGBSpectrum f_c = suncol * bsdf.f(wo, normalize(sunpos), primary_surface)
					* dot(payload.w_shading_norm, normalize(sunpos)) * t_globals.IntegratorCFG.sunlight_intensity;
				light += f_c * throughtput;
			}
		}

		BSDFSample bs = bsdf.sampleBSDF(wo, Samplers::get2D_PCGHash(t_seed), primary_surface);
		float3 wi = bs.wi;
		float pdf = bs.pdf;
		if (primary_surface)bs.f = RGBSpectrum(1);
		RGBSpectrum fcos = bs.f * AbsDot(wi, payload.w_shading_norm);
		if (!fcos)break;
		throughtput *= fcos / pdf;

		p_b = bs.pdf;
		prev_ctx = LightSampleContext(payload);

		ray = payload.spawnRay(wi);

		RGBSpectrum RR_beta = throughtput * eta_scale;
		if (RR_beta.maxComponentValue() < 1 && bounce_depth > 1) {
			float q = fmaxf(0.f, 1.f - RR_beta.maxComponentValue());
			if (Samplers::get1D_PCGHash(t_seed) < q)
				break;
			throughtput /= 1 - q;
		}
	}

	return clampOutput(light);
}

__device__ RGBSpectrum IntegratorPipeline::SampleLd(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& payload,
	const BSDF& bsdf, const LightSampler& light_sampler, uint32_t& seed, bool primary_surface)
{
	RGBSpectrum Ld(0.f);
	SampledLight sampled_light = light_sampler.sample(Samplers::get1D_PCGHash(seed));

	//handle empty buffer
	if (!sampled_light)return Ld;

	LightLiSample ls = sampled_light.light->SampleLi(globals, payload, Samplers::get2D_PCGHash(seed));

	if (ls.pdf <= 0)return Ld;

	float3 wi = ls.wi;
	float3 wo = -ray.getDirection();
	RGBSpectrum f = ((primary_surface) ? RGBSpectrum(1) : bsdf.f(wo, wi, primary_surface)) * AbsDot(wi, payload.w_shading_norm);

	if (!f || !Unoccluded(globals, payload, ls.pLight)) return Ld;

	float dist = length(payload.w_pos - ls.pLight);
	float dist_sq = dist * dist;
	float cosTheta_emitter = AbsDot(wi, ls.n);
	float Li_sample_pdf = (sampled_light.p * ls.pdf) * (1 / cosTheta_emitter) * dist_sq;
	float p_l = Li_sample_pdf;
	float p_b = bsdf.pdf(wo, wi);
	float w_l = powerHeuristic(1, p_l, 1, p_b);

	Ld = w_l * f * ls.L / p_l;
	return Ld;
};

__device__ RGBSpectrum staticAccumulation(const IntegratorGlobals& globals, RGBSpectrum radiance_sample, int2 c_pix) {
	float s = getLuminance(radiance_sample);
	float s2 = s * s;
	globals.FrameBuffer.variance_accumulation_framebuffer
		[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] += make_float3(s, s2, 0);
	float3 avg_mom = (
		globals.FrameBuffer.variance_accumulation_framebuffer[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] / (globals.FrameIndex)
		);

	float var = fabsf(avg_mom.y - Sqr(avg_mom.x));

	texWrite(make_float4(make_float3(var), 1),
		globals.FrameBuffer.svgf_filtered_variance_front_surfobject,
		c_pix);

	globals.FrameBuffer.accumulation_framebuffer
		[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] += make_float3(radiance_sample);
	return RGBSpectrum(
		globals.FrameBuffer.accumulation_framebuffer[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] / (globals.FrameIndex)
	);
}

//must write to irradiance
__global__ void tracePathSample(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;

	//--------------------------------------------------

	uint32_t seed = texReadNearest(t_globals.FrameBuffer.seeds_surfobject, current_pix).x;

	//RGBSpectrum sampled_radiance = IntegratorPipeline::evaluatePixelSample(t_globals, make_float2(current_pix));

	RGBSpectrum sampled_radiance = IntegratorPipeline::deferredEvaluatePixelSample(t_globals, current_pix, seed);

	float4 current_radiance = make_float4(sampled_radiance, 1);

	texWrite(current_radiance,
		t_globals.FrameBuffer.raw_irradiance_surfobject, current_pix);
}