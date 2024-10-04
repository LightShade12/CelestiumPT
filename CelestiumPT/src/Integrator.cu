#include "Integrator.cuh"
#include "Film.cuh"
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
#include "maths/Sampling.cuh"
#include "maths/constants.cuh"

#include <device_launch_parameters.h>
#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <float.h>

__host__ void IntegratorPipeline::invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims)
{
	renderKernel << < block_grid_dims, thread_block_dims >> > (globals);
};

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
	return IntegratorPipeline::LiPathIntegrator(globals, ray, seed, ppixel);
}

__device__ RGBSpectrum IntegratorPipeline::LiPathIntegrator(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel)
{
	Ray ray = in_ray;
	RGBSpectrum throughtput(1.f), light(0.f);
	LightSampler light_sampler(
		globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsBuffer,
		globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsCount);
	float p_b = 1;
	LightSampleContext prev_ctx{};
	ShapeIntersection payload{};
	float eta_scale = 1;//TODO: look up russian roulette

	for (int bounce_depth = 0; bounce_depth <= globals.IntegratorCFG.max_bounces; bounce_depth++) {
		seed += bounce_depth;

		payload = IntegratorPipeline::Intersect(globals, ray);

		bool primary_surface = (bounce_depth == 0);

		if (primary_surface) recordGBufferAny(globals, ppixel, payload);

		//miss--
		if (payload.hit_distance < 0)//TODO: standardize invalid/miss payload definition
		{
			if (primary_surface) recordGBufferMiss(globals, ppixel);

			light += globals.SceneDescriptor.device_geometry_aggregate->SkyLight.Le(ray) * throughtput;
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
				float dist = length(prev_ctx.pos - payload.w_pos);
				float cosTheta_emitter = AbsDot(-ray.getDirection(), payload.w_geo_norm);
				light_pdf = light_pdf * (1.f / cosTheta_emitter) * Sqr(dist);
				float w_l = powerHeuristic(1, p_b, 1, light_pdf);
				light += Le * throughtput * w_l;
			}
		}

		BSDF bsdf = payload.getBSDF(globals);

		RGBSpectrum Ld = SampleLd(globals, ray, payload, bsdf,
			light_sampler, seed, primary_surface);
		light += Ld * throughtput;

		BSDFSample bs = bsdf.sampleBSDF(wo, Samplers::get2D_PCGHash(seed));
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
	RGBSpectrum f = ((primary_surface) ? RGBSpectrum(1) : bsdf.f(wo, wi)) * AbsDot(wi, payload.w_shading_norm);

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

__device__ bool rejectionHeuristic(const IntegratorGlobals& globals, int2 prev_pix, int2 cur_px) {
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

__device__ float4 sampleBilinear(const IntegratorGlobals& globals, const cudaSurfaceObject_t& tex_surface,
	float2 fpix, bool lerp_alpha)
{
	//TODO:consider half pixel for centre smapling
	// Integer pixel coordinates
	int2 pix = make_int2(fpix);
	int x = pix.x;
	int y = pix.y;

	// Get resolution
	int2 res = globals.FrameBuffer.resolution;

	// Clamp pixel indices to be within bounds
	int s0 = clamp(x, 0, res.x - 1);
	int s1 = clamp(x + 1, 0, res.x - 1);
	int t0 = clamp(y, 0, res.y - 1);
	int t1 = clamp(y + 1, 0, res.y - 1);

	// Compute fractional parts for interpolation weights
	float ws = fpix.x - s0;
	float wt = fpix.y - t0;

	// Sample 2x2 texel neighborhood
	float4 cp0 = surf2Dread<float4>(tex_surface, s0 * (int)sizeof(float4), t0);
	float4 cp1 = surf2Dread<float4>(tex_surface, s1 * (int)sizeof(float4), t0);
	float4 cp2 = surf2Dread<float4>(tex_surface, s0 * (int)sizeof(float4), t1);
	float4 cp3 = surf2Dread<float4>(tex_surface, s1 * (int)sizeof(float4), t1);

	// Perform bilinear interpolation
	float4 tc0 = cp0 + (cp1 - cp0) * ws;
	float4 tc1 = cp2 + (cp3 - cp2) * ws;
	float4 fc = tc0 + (tc1 - tc0) * wt;

	// Handle alpha channel based on lerp_alpha flag
	if (!lerp_alpha) {
		// Nearest neighbor for alpha
		fc.w = (ws > 0.5f ? (wt > 0.5f ? cp3.w : cp1.w) : (wt > 0.5f ? cp2.w : cp0.w));
	}

	return fc;
}

__device__ void computeVelocity(const IntegratorGlobals& globals, float2 tc_uv, int2 ppixel) {
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

	float2 c_uv = (make_float2(c_ndc) + 1.f) / 2.f;//0->1
	float2 p_uv = (make_float2(p_ndc) + 1.f) / 2.f;

	float2 vel = c_uv - p_uv;

	float3 velcol = make_float3(0);

	//velcol += (vel.x > 0) ? make_float3(vel.x, 0, 0) : make_float3(0, fabsf(vel.x), fabsf(vel.x));
	//velcol += (vel.y > 0) ? make_float3(0, vel.y, 0) : make_float3(fabsf(vel.y), 0, fabsf(vel.y));
	velcol = make_float3(vel, 0);

	surf2Dwrite(make_float4(velcol, 1),
		globals.FrameBuffer.velocity_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
}

__device__ RGBSpectrum temporalAccumulation(const IntegratorGlobals& globals, RGBSpectrum c_col, float2 c_uv, int2 c_pix) {
	RGBSpectrum final_color = c_col;
	float4 c_objID = surf2Dread<float4>(globals.FrameBuffer.objectID_render_surface_object,
		c_pix.x * (int)sizeof(float4), c_pix.y);
	int objID = c_objID.x;

	//void sample/ miss/ sky
	if (objID < 0) {
		surf2Dwrite<float4>(make_float4(final_color, 0), globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			c_pix.x * (int)sizeof(float4), c_pix.y);
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
		surf2Dwrite<float4>(make_float4(final_color, 0), globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			c_pix.x * (int)sizeof(float4), c_pix.y);
		return final_color;
	}

	bool prj_success = !rejectionHeuristic(globals, prev_px, c_pix);

	//disocclusion/ reproj failure
	if (!prj_success) {
		surf2Dwrite<float4>(make_float4(final_color, 0), globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			c_pix.x * (int)sizeof(float4), c_pix.y);
		return final_color;
	}

	float4 hist_col = sampleBilinear(globals,
		globals.FrameBuffer.history_integrated_irradiance_front_surfobj, prev_pxf, false);
	//float4 hist_col = surf2Dread<float4>(globals.FrameBuffer.history_integrated_irradiance_front_surfobj,
	//	prev_px.x * (int)sizeof(float4), prev_px.y);
	float hist_len = hist_col.w;

	const int MAX_ACCUMULATION_FRAMES = 16;
	final_color = RGBSpectrum(lerp(make_float3(hist_col), make_float3(c_col),
		1.f / fminf(float(hist_len + 1), MAX_ACCUMULATION_FRAMES)));

	//feedback
	surf2Dwrite<float4>(make_float4(final_color, hist_len + 1),
		globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
		c_pix.x * (int)sizeof(float4), c_pix.y);
	return final_color;
}

__device__ RGBSpectrum staticAccumulation(const IntegratorGlobals& globals, RGBSpectrum radiance_sample, int2 c_pix) {
	globals.FrameBuffer.accumulation_framebuffer
		[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] += make_float3(radiance_sample);
	return RGBSpectrum(
		globals.FrameBuffer.accumulation_framebuffer[c_pix.x + c_pix.y * globals.FrameBuffer.resolution.x] / (globals.frameidx)
	);
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

	if (globals.IntegratorCFG.temporal_accumulation) {
		sampled_radiance = temporalAccumulation(globals, sampled_radiance, screen_uv, ppixel);
	}
	else if (globals.IntegratorCFG.accumulate) {
		sampled_radiance = staticAccumulation(globals, sampled_radiance, ppixel);
	}

	float4 sampled_albedo = surf2Dread<float4>(globals.FrameBuffer.albedo_render_surface_object,
		thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);

	sampled_radiance *= RGBSpectrum(sampled_albedo);//MODULATE

	RGBSpectrum frag_spectrum = sampled_radiance;
	//EOTF
	frag_spectrum = gammaCorrection(frag_spectrum);

	frag_spectrum = toneMapping(frag_spectrum, 8);

	//frag_spectrum *= 3.3f;
	//frag_spectrum = agx_fitted(frag_spectrum);
	//frag_spectrum = agx_fitted_Eotf(frag_spectrum);

	//frag_spectrum = agx_tonemapping(frag_spectrum);

	float4 frag_color = make_float4(frag_spectrum, 1);

	surf2Dwrite(frag_color, globals.FrameBuffer.composite_render_surface_object,
		thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}