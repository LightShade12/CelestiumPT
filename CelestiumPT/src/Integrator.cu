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
#include <device_functions.h>
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

	RGBSpectrum W(11.2f);
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

__device__ bool rejectionHeuristic(const IntegratorGlobals& globals, int2 prev_pix, int2 cur_px) {
	float4 p_depth = surf2Dread<float4>(globals.FrameBuffer.history_depth_render_surface_object,
		prev_pix.x * (int)sizeof(float4), prev_pix.y);
	float p_sampled_depth = p_depth.x;

	float3 p_cpos = make_float3(globals.SceneDescriptor.active_camera->prev_viewMatrix.inverse() * make_float4(0, 0, 0, 1));

	//-------
	float4 c_lpos = surf2Dread<float4>(globals.FrameBuffer.local_positions_render_surface_object,
		cur_px.x * (int)sizeof(float4), cur_px.y);

	float4 c_objID = surf2Dread<float4>(globals.FrameBuffer.objectID_render_surface_object,
		cur_px.x * (int)sizeof(float4), cur_px.y);
	float3 l_pos = make_float3(c_lpos);
	int objID = c_objID.x;

	Mat4 p_M = globals.SceneDescriptor.device_geometry_aggregate->DeviceMeshesBuffer[objID].prev_modelMatrix;

	//-------------
	float3 p_wpos = make_float3(p_M * make_float4(l_pos, 1));//clipspace

	float estimated_depth = length(p_cpos - p_wpos);

	float TEMP_DEPTH_REJECT_THRESHOLD = 0.01f;

	if (fabsf(estimated_depth - p_sampled_depth) > (p_sampled_depth * TEMP_DEPTH_REJECT_THRESHOLD)) {
		return true;
	}
	//------------
	float4 p_wnorm = surf2Dread<float4>(globals.FrameBuffer.history_world_normals_render_surface_object,
		prev_pix.x * (int)sizeof(float4), prev_pix.y);
	float3 p_sampled_wnorm = normalize(make_float3(p_wnorm));

	float4 c_lnorm = surf2Dread<float4>(globals.FrameBuffer.local_normals_render_surface_object,
		prev_pix.x * (int)sizeof(float4), prev_pix.y);
	float3 c_sampled_lnorm = make_float3(c_lnorm);

	float3 estimated_wnorm = normalize(make_float3(p_M * make_float4(c_sampled_lnorm, 0)));

	float TEMP_NORMALS_REJECT_THRESHOLD = fabsf(cosf(deg2rad(45)));//TODO:make consexpr

	if (AbsDot(p_sampled_wnorm, estimated_wnorm) < TEMP_NORMALS_REJECT_THRESHOLD) {
		return true;
	}

	return false;
}

__device__ RGBSpectrum temporalAccumulation(const IntegratorGlobals& globals, RGBSpectrum c_col, float2 c_uv, int2 ppixel) {
	RGBSpectrum final_color = c_col;
	float4 c_objID = surf2Dread<float4>(globals.FrameBuffer.objectID_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	int objID = c_objID.x;

	//void sample/ miss/ sky
	if (objID < 0) {
		surf2Dwrite<float4>(make_float4(final_color, 0), globals.FrameBuffer.history_color_render_back_surface_object,
			ppixel.x * (int)sizeof(float4), ppixel.y);
		return final_color;
	}

	float4 c_vel = surf2Dread<float4>(globals.FrameBuffer.velocity_render_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);

	//reproject
	float2 vel = make_float2(c_vel.x, c_vel.y);//ndc 0->+-2
	//float2 pixel_offset = (vel / 2.f) * make_float2(globals.FrameBuffer.resolution);
	float2 pixel_offset = (vel)*make_float2(globals.FrameBuffer.resolution);
	int2 prev_px = ppixel - make_int2(pixel_offset);

	//new fragment
	if (prev_px.x < 0 || prev_px.x >= globals.FrameBuffer.resolution.x ||
		prev_px.y < 0 || prev_px.y >= globals.FrameBuffer.resolution.y) {
		surf2Dwrite<float4>(make_float4(final_color, 0), globals.FrameBuffer.history_color_render_back_surface_object,
			ppixel.x * (int)sizeof(float4), ppixel.y);
		return final_color;
	}

	bool prj_success = !rejectionHeuristic(globals, prev_px, ppixel);

	//disocclusion/ reproj failure
	if (!prj_success) {
		surf2Dwrite<float4>(make_float4(final_color, 0), globals.FrameBuffer.history_color_render_back_surface_object,
			ppixel.x * (int)sizeof(float4), ppixel.y);
		return final_color;
	}

	//TODO: try to use 3x3 bilinear texel sampling
	float4 hist_col = surf2Dread<float4>(globals.FrameBuffer.history_color_render_front_surface_object,
		prev_px.x * (int)sizeof(float4), prev_px.y);
	float hist_len = hist_col.w;

	const int MAX_ACCUMULATION_FRAMES = 16;
	final_color = RGBSpectrum(lerp(make_float3(hist_col), make_float3(c_col),
		1.f / fminf(float(hist_len + 1), MAX_ACCUMULATION_FRAMES)));

	//feedback
	surf2Dwrite<float4>(make_float4(final_color, hist_len + 1),
		globals.FrameBuffer.history_color_render_back_surface_object,
		ppixel.x * (int)sizeof(float4), ppixel.y);
	return final_color;
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

	float2 vel = (c_uv)-(p_uv);

	float3 velcol = make_float3(0);

	//velcol += (vel.x > 0) ? make_float3(vel.x, 0, 0) : make_float3(0, fabsf(vel.x), fabsf(vel.x));
	//velcol += (vel.y > 0) ? make_float3(0, vel.y, 0) : make_float3(fabsf(vel.y), 0, fabsf(vel.y));
	velcol = make_float3(vel, 0);

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

	if (!globals.IntegratorCFG.temporal_accumulation) {
		sampled_radiance = temporalAccumulation(globals, sampled_radiance, screen_uv, ppixel);
	}
	else if (globals.IntegratorCFG.accumulate) {
		globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] += make_float3(sampled_radiance);
		sampled_radiance = RGBSpectrum(globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] / (globals.frameidx));
	}

	RGBSpectrum frag_spectrum = sampled_radiance;
	//EOTF
	frag_spectrum = gammaCorrection(frag_spectrum);
	frag_spectrum = toneMapping(frag_spectrum, 8);
	float4 frag_color = make_float4(frag_spectrum, 1);

	surf2Dwrite(frag_color, globals.FrameBuffer.composite_render_surface_object, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
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
	return (1.0f - a) * RGBSpectrum(1.0, 1.0, 1.0) + a * RGBSpectrum(0.2, 0.4, 1.0);
};

__device__ void recordGBufferHit(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si) {
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

__device__ void recordGBufferAny(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si) {
	//float2 uv = ppixel / make_float2(globals.FrameBuffer.resolution);
	//float3 dbg_uv_col = make_float3(uv);

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

__device__ float balanceHeuristic(int nf, float fPdf, int ng, float gPdf) {
	return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

__device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
	float f = nf * fPdf, g = ng * gPdf;
	return Sqr(f) / (Sqr(f) + Sqr(g));
}

__device__ static bool checkNaN(const float3& vec) {
	return isnan(vec.x) || isnan(vec.y) || isnan(vec.z);
}
__device__ static bool checkINF(const float3& vec) {
	return isinf(vec.x) || isinf(vec.y) || isinf(vec.z);
}

__device__ RGBSpectrum clampOutput(const RGBSpectrum& rgb) {
	if ((checkNaN(make_float3(rgb))) || (checkINF(make_float3(rgb))))
		return RGBSpectrum(0);
	else
		return RGBSpectrum(clamp(make_float3(rgb), 0, 1000));
}

__device__ RGBSpectrum IntegratorPipeline::LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel)
{
	Ray ray = in_ray;
	//bool DI = false;
	RGBSpectrum throughtput(1.f), light(0.f);
	LightSampler light_sampler(
		globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsBuffer,
		globals.SceneDescriptor.device_geometry_aggregate->DeviceLightsCount);
	float p_b = 1;
	LightSampleContext prev_ctx{};
	ShapeIntersection payload{};

	for (int bounce_depth = 0; bounce_depth <= globals.IntegratorCFG.max_bounces; bounce_depth++) {
		seed += bounce_depth;

		payload = IntegratorPipeline::Intersect(globals, ray);

		bool primary_surface = (bounce_depth == 0);

		if (primary_surface) recordGBufferAny(globals, ppixel, payload);

		//miss--
		if (payload.hit_distance < 0)//TODO: standardize invalid/miss payload definition
		{
			if (primary_surface) recordGBufferMiss(globals, ppixel);

			light += SkyShading(ray) * throughtput * 0.f;
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

		//get BSDF
		BSDF bsdf = payload.getBSDF(globals);

		RGBSpectrum Ld = SampleLd(globals, ray, payload, bsdf, light_sampler, seed);
		light += Ld * throughtput;

		BSDFSample bs = bsdf.sampleBSDF(wo, Samplers::get2D_PCGHash(seed));
		float3 wi = bs.wi;
		float pdf = bs.pdf;
		RGBSpectrum fcos = bs.f * AbsDot(wi, payload.w_shading_norm);
		if (!fcos)break;
		throughtput *= fcos / pdf;

		p_b = bs.pdf;
		prev_ctx = LightSampleContext(payload);

		ray = payload.spawnRay(wi);
	}

	return clampOutput(light);
}
__device__ RGBSpectrum IntegratorPipeline::SampleLd(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& payload,
	const BSDF& bsdf, const LightSampler& light_sampler, uint32_t& seed)
{
	RGBSpectrum Ld(0.f);
	SampledLight sampled_light = light_sampler.sample(Samplers::get1D_PCGHash(seed));

	//handle empty buffer
	if (!sampled_light)return Ld;

	LightLiSample ls = sampled_light.light->SampleLi(payload, Samplers::get2D_PCGHash(seed));

	if (ls.pdf <= 0)return Ld;

	float3 wi = ls.wi;
	float3 wo = -ray.getDirection();
	RGBSpectrum f = bsdf.f(wo, wi) * AbsDot(wi, payload.w_shading_norm);
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