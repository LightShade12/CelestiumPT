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
#include "temporal_pass.cuh"

#include "maths/linear_algebra.cuh"
#include "maths/sampling.cuh"
#include "maths/constants.cuh"

#include <device_launch_parameters.h>
#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <float.h>

__device__ RGBSpectrum IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel)
{
	uint32_t seed = ppixel.x + ppixel.y * globals.FrameBuffer.resolution.x;
	seed *= globals.FrameIndex;

	int2 frameres = globals.FrameBuffer.resolution;

	float2 screen_uv = { (ppixel.x / frameres.x),(ppixel.y / frameres.y) };
	screen_uv = screen_uv * 2 - 1;//-1->1

	Ray primary_ray = globals.SceneDescriptor.ActiveCamera->generateRay(frameres.x, frameres.y, screen_uv);

	RGBSpectrum L = IntegratorPipeline::Li(globals, primary_ray, seed, ppixel);

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

__device__ RGBSpectrum IntegratorPipeline::Li(const IntegratorGlobals& globals, const Ray& ray, uint32_t seed, float2 ppixel)
{
	return IntegratorPipeline::LiPathIntegrator(globals, ray, seed, ppixel);
}

__device__ RGBSpectrum IntegratorPipeline::LiPathIntegrator(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel)
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
	//float3 sunpos = make_float3(sinf(globals.FrameIndex * 0.01f), 1, cosf(globals.FrameIndex * 0.01f)) * 100;
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

			//light += globals.SceneDescriptor.DeviceGeometryAggregate->SkyLight.Le(ray) * RGBSpectrum(0.8, 1, 1.5) * throughtput;
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

__device__ RGBSpectrum IntegratorPipeline::SampleLd(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& payload,
	const BSDF& bsdf, const LightSampler& light_sampler, uint32_t& seed, bool primary_surface)
{
	RGBSpectrum Ld(0.f);
	SampledLight sampled_light = light_sampler.sample(Samplers::get1D_PCGHash(seed));

	//handle empty buffer
	if (!sampled_light)return Ld;

	LightLiSample ls = sampled_light.light->SampleLi(payload, Samplers::get2D_PCGHash(seed));

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

//Accumulation-----------------------------------------------------

//velocity is in screen UV space
__device__ void computeVelocity(const IntegratorGlobals& globals, float2 tc_uv, int2 ppixel) {
	float4 c_lpos = texReadNearest(globals.FrameBuffer.local_positions_render_surface_object, ppixel);
	float4 c_objID = texReadNearest(globals.FrameBuffer.objectID_render_surface_object, ppixel);

	float3 l_pos = make_float3(c_lpos);
	float objID = c_objID.x;

	if (objID < 0) {
		texWrite(make_float4(0, 0, 0, 1),
			globals.FrameBuffer.velocity_render_surface_object, ppixel);
		return;
	}

	Mat4 c_VP = globals.SceneDescriptor.ActiveCamera->projectionMatrix * globals.SceneDescriptor.ActiveCamera->viewMatrix;
	Mat4 p_VP = globals.SceneDescriptor.ActiveCamera->prev_projectionMatrix *
		globals.SceneDescriptor.ActiveCamera->prev_viewMatrix;
	Mat4 c_M = globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMeshesBuffer[(int)objID].modelMatrix;
	Mat4 p_M = globals.SceneDescriptor.DeviceGeometryAggregate->DeviceMeshesBuffer[(int)objID].prev_modelMatrix;

	float4 c_inpos = c_VP * c_M * make_float4(l_pos, 1);//clipspace
	float4 p_inpos = p_VP * p_M * make_float4(l_pos, 1);//clipspace

	float3 c_ndc = make_float3(c_inpos) / c_inpos.w;
	float3 p_ndc = make_float3(p_inpos) / p_inpos.w;

	float2 c_uv = (make_float2(c_ndc) + 1.f) / 2.f;//0->1
	float2 p_uv = (make_float2(p_ndc) + 1.f) / 2.f;

	float2 vel = c_uv - p_uv;

	float3 velcol = make_float3(0);

	velcol = make_float3(vel, 0);

	texWrite(make_float4(velcol, 1),
		globals.FrameBuffer.velocity_render_surface_object, ppixel);
}

__device__ RGBSpectrum temporalAccumulation(const IntegratorGlobals& globals, RGBSpectrum c_col, float4 c_moments, float2 c_uv, int2 c_pix) {
	RGBSpectrum final_color = c_col;
	float4 c_objID = surf2Dread<float4>(globals.FrameBuffer.objectID_render_surface_object,
		c_pix.x * (int)sizeof(float4), c_pix.y);
	int objID = c_objID.x;
	//float4 sampled_moments = texReadNearest(globals.FrameBuffer.current_moments_render_surface_object, c_pix);
	float4 sampled_moments = c_moments;
	float2 final_moments = make_float2(sampled_moments.x, sampled_moments.y);

	//void sample/ miss/ sky
	if (objID < 0) {
		float variance = fabsf(final_moments.y - Sqr(final_moments.x));
		texWrite(make_float4(make_float3(variance), 1),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			c_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			c_pix);
		texWrite(make_float4(final_color, 0),
			globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			c_pix);
		return final_color;
	}

	float4 c_vel = surf2Dread<float4>(globals.FrameBuffer.velocity_render_surface_object,
		c_pix.x * (int)sizeof(float4), c_pix.y);

	//reproject
	float2 vel = make_float2(c_vel.x, c_vel.y);
	float2 pixel_offset = (vel)*make_float2(globals.FrameBuffer.resolution);
	int2 prev_px = c_pix - make_int2(pixel_offset);
	float2 prev_pxf = make_float2(c_pix) - pixel_offset;

	//new fragment
	if (prev_px.x < 0 || prev_px.x >= globals.FrameBuffer.resolution.x ||
		prev_px.y < 0 || prev_px.y >= globals.FrameBuffer.resolution.y) {
		float variance = fabsf(final_moments.y - Sqr(final_moments.x));
		texWrite(make_float4(make_float3(variance), 1),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			c_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			c_pix);
		texWrite(make_float4(final_color, 0), globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			c_pix);
		return final_color;
	}

	bool prj_success = !rejectionHeuristic(globals, prev_px, c_pix);

	//disocclusion/ reproj failure
	if (!prj_success) {
		float variance = fabsf(final_moments.y - Sqr(final_moments.x));
		texWrite(make_float4(make_float3(variance), 1),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			c_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			c_pix);
		texWrite(make_float4(final_color, 0),
			globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			c_pix);
		return final_color;
	}
	const int MAX_ACCUMULATION_FRAMES = 16;

	float4 hist_col = texReadBilinear(globals.FrameBuffer.history_integrated_irradiance_front_surfobj, prev_pxf,
		globals.FrameBuffer.resolution, false);
	float4 hist_moments = texReadNearest(globals.FrameBuffer.history_integrated_moments_front_surfobj, prev_px);

	float moments_hist_len = hist_moments.w;
	float hist_len = hist_col.w;

	final_moments = lerp(make_float2(hist_moments.x, hist_moments.y), make_float2(final_moments.x, final_moments.y),
		1.f / fminf(float(moments_hist_len + 1), MAX_ACCUMULATION_FRAMES));

	final_color = RGBSpectrum(lerp(make_float3(hist_col), make_float3(c_col),
		1.f / fminf(float(hist_len + 1), MAX_ACCUMULATION_FRAMES)));

	float variance = fabsf(final_moments.y - Sqr(final_moments.x));

	//variamce
	texWrite(make_float4(make_float3(variance), 1),
		globals.FrameBuffer.filtered_variance_render_front_surfobj,
		c_pix);

	//feedback: moments
	texWrite(make_float4(final_moments.x, final_moments.y, 0, moments_hist_len + 1),
		globals.FrameBuffer.history_integrated_moments_back_surfobj,
		c_pix);

	//feedback
	texWrite(make_float4(final_color, hist_len + 1),
		globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
		c_pix);

	return final_color;
}

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
		globals.FrameBuffer.filtered_variance_render_front_surfobj,
		c_pix);

	globals.FrameBuffer.accumulation_framebuffer
		[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] += make_float3(radiance_sample);
	return RGBSpectrum(
		globals.FrameBuffer.accumulation_framebuffer[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] / (globals.FrameIndex)
	);
}

//must write to irradiance, moment data & gbuffer
__global__ void renderPathTraceRaw(const IntegratorGlobals globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;

	//--------------------------------------------------

	RGBSpectrum sampled_radiance = IntegratorPipeline::evaluatePixelSample(globals, make_float2(current_pix));

	float s = getLuminance(sampled_radiance);
	float s2 = s * s;
	float4 current_moments = make_float4(s, s2, 0, 1);

	computeVelocity(globals, screen_uv, current_pix);//this concludes all Gbuffer data writes

	float4 current_radiance = make_float4(sampled_radiance, 1);

	texWrite(current_radiance,
		globals.FrameBuffer.current_irradiance_render_surface_object, current_pix);
	texWrite(current_moments,
		globals.FrameBuffer.current_moments_render_surface_object, current_pix);
}