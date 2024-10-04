#include "svgf.cuh"

#include "Integrator.cuh"
#include "Spectrum.cuh"
#include "Storage.cuh"
#include "maths/maths_linear_algebra.cuh"
#include "Film.cuh"
#include "ErrorCheck.cuh"

#define __CUDACC__
#include <device_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Structure representing the G-buffer data
struct GBuffer {
	float3 irradiance;
	float3 normal;
	float  depth;
	float  variance;
};
__device__ float getLuminance(const RGBSpectrum& col) {
	// Rec. 709 luminance coefficients for linear RGB
	return 0.2126f * col.r + 0.7152f * col.g + 0.0722f * col.b;
}
__device__ float4 texRead(cudaSurfaceObject_t tex_surf, int2 pix) {
	float4 data = surf2Dread<float4>(tex_surf,
		pix.x * (int)sizeof(float4), pix.y);
	return data;
}
//has to be uchar4/2/1 or float4/2/1; no 3 comp color
__device__ void texWrite(float4 data, cudaSurfaceObject_t tex_surf, int2 pix) {
	surf2Dwrite<float4>(data, tex_surf, pix.x * (int)sizeof(float4), pix.y);
}

//must write to irradiance, moment data & gbuffer
__global__ void renderPathTraceRaw(IntegratorGlobals globals)
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

	computeVelocity(globals, screen_uv, current_pix);//this concludes all Gbuffer data writes

	//if (globals.IntegratorCFG.temporal_accumulation) {
	//	sampled_radiance = temporalAccumulation(globals, sampled_radiance, screen_uv, current_pix);
	//}
	//else if (globals.IntegratorCFG.accumulate) {
	//	sampled_radiance = staticAccumulation(globals, sampled_radiance, current_pix);
	//}

	//float4 sampled_albedo = texRead(globals.FrameBuffer.albedo_render_surface_object, current_pix);
	//
	//sampled_radiance *= RGBSpectrum(sampled_albedo);//MODULATE
	//
	//RGBSpectrum frag_spectrum = sampled_radiance;
	//EOTF
	//frag_spectrum = gammaCorrection(frag_spectrum);
	//frag_spectrum = toneMapping(frag_spectrum, 8);
	//
	//float4 frag_color = make_float4(frag_spectrum, 1);

	float s = getLuminance(sampled_radiance);
	float s2 = s * s;
	float4 current_moments = make_float4(s, s2, 0, 0);
	float4 current_radiance = make_float4(sampled_radiance, 1);

	texWrite(current_radiance,
		globals.FrameBuffer.current_irradiance_render_surface_object, current_pix);
	texWrite(current_moments,
		globals.FrameBuffer.current_moments_render_surface_object, current_pix);
}

//feedback only moments
__global__ void temporalIntegrate(IntegratorGlobals globals) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texRead(globals.FrameBuffer.current_irradiance_render_surface_object, current_pix);
	float4 sampled_moments = texRead(globals.FrameBuffer.current_moments_render_surface_object, current_pix);

	RGBSpectrum final_irradiance = RGBSpectrum(sampled_irradiance);
	float2 final_moments = make_float2(sampled_moments.x, sampled_moments.y);

	int current_objID = texRead(globals.FrameBuffer.objectID_render_surface_object, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no accumulate

		float variance = fabsf(final_moments.y - Sqr(final_moments.x));
		texWrite(make_float4(make_float3(variance), 0),
			globals.FrameBuffer.variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 1),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);

		return;
	}

	float4 current_vel = texRead(globals.FrameBuffer.velocity_render_surface_object, current_pix);

	//reproject
	int2 res = globals.FrameBuffer.resolution;
	float2 velf2 = make_float2(current_vel.x, current_vel.y);
	float2 pixel_offset = (velf2)*make_float2(res);

	int2 prev_px = current_pix - make_int2(pixel_offset);
	float2 prev_pxf = make_float2(current_pix) - pixel_offset;

	//new fragment; out of screen
	if (prev_px.x < 0 || prev_px.x >= res.x ||
		prev_px.y < 0 || prev_px.y >= res.y) {
		float variance = fabsf(final_moments.y - Sqr(final_moments.x));
		texWrite(make_float4(make_float3(variance), 0),
			globals.FrameBuffer.variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 1),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);

		return;
	}

	if (prev_px.y < 0)printf(" illegal pixel!");

	bool prj_success = !rejectionHeuristic(globals, prev_px, current_pix);

	//disocclusion/ reproj failure
	if (!prj_success) {
		float variance = fabsf(final_moments.y - Sqr(final_moments.x));
		texWrite(make_float4(make_float3(variance), 0),
			globals.FrameBuffer.variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
			current_pix);
		texWrite(make_float4(final_irradiance, 1),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		return;
	}

	float4 hist_irradiance = sampleBilinear(globals,
		globals.FrameBuffer.history_integrated_irradiance_front_surfobj, prev_pxf, false);
	float4 hist_moments = texRead(globals.FrameBuffer.history_integrated_moments_front_surfobj, prev_px);

	float irradiance_hist_len = hist_irradiance.w;
	float moments_hist_len = hist_moments.w;

	const int MAX_ACCUMULATION_FRAMES = 16;

	final_irradiance = RGBSpectrum(
		lerp(make_float3(hist_irradiance), make_float3(final_irradiance),
			1.f / fminf(float(irradiance_hist_len + 1), MAX_ACCUMULATION_FRAMES))
	);

	final_moments = lerp(make_float2(hist_moments.x, hist_moments.y), make_float2(final_moments.x, final_moments.y),
		1.f / fminf(float(moments_hist_len + 1), MAX_ACCUMULATION_FRAMES));

	float variance = fabsf(final_moments.y - Sqr(final_moments.x));
	texWrite(make_float4(make_float3(variance), 0),
		globals.FrameBuffer.variance_render_front_surfobj,
		current_pix);

	//feedback: moments
	texWrite(make_float4(final_moments.x, final_moments.y, 0, moments_hist_len + 1),
		globals.FrameBuffer.history_integrated_moments_back_surfobj,
		current_pix);

	//write integrated irradiance
	texWrite(make_float4(final_irradiance, irradiance_hist_len + 1),
		globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
}

// Normal-weighting function (4.4.1)
__device__ float normalWeight(float3 normal0, float3 normal1) {
	const float exponent = 64.0;
	return pow(max(0.0, dot(normal0, normal1)), exponent);
}

// Depth-weighting function (4.4.2)
__device__ float depthWeight(float depth0, float depth1, float2 grad, float2 offset) {
	// paper uses eps = 0.005 for a normalized depth buffer
	// ours is not but 0.1 seems to work fine
	const float eps = 0.1;
	return exp((-abs(depth0 - depth1)) / (abs(dot(grad, offset)) + eps));
}

// Luminance-weighting function (4.4.3)
__device__ float luminanceWeight(float lum0, float lum1, float variance) {
	const float strictness = 4.0;
	const float eps = 0.01;
	return exp((-abs(lum0 - lum1)) / (strictness * variance + eps));
}

__device__ GBuffer sampleGBuffer(const IntegratorGlobals& globals, int2 c_pix) {
	GBuffer g;
	g.irradiance = make_float3(texRead(globals.FrameBuffer.filtered_irradiance_front_render_surface_object, c_pix));
	g.variance = texRead(globals.FrameBuffer.variance_render_front_surfobj, c_pix).x;
	g.normal = make_float3(texRead(globals.FrameBuffer.world_normals_render_surface_object, c_pix));
	g.depth = texRead(globals.FrameBuffer.depth_render_surface_object, c_pix).x;
	return g;
}

__device__ void writeGBuffer(const IntegratorGlobals& globals, const GBuffer& gbuffer, int2 c_pix) {
	texWrite(make_float4(gbuffer.irradiance, 1),
		globals.FrameBuffer.filtered_irradiance_back_render_surface_object, c_pix);
	texWrite(make_float4(make_float3(gbuffer.variance), 1),
		globals.FrameBuffer.variance_render_back_surfobj, c_pix);
}

//ensure this is not oob
__device__ float4 dFdx(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res) {
	float4 d0 = texRead(data_surfobj, c_pix);
	int2 next = c_pix + make_int2(1, 0);
	next.x = clamp(next.x, 0, res.x - 1);
	float4 d1 = texRead(data_surfobj, next);
	return d1 - d0;
}

//ensure this is not oob
__device__ float4 dFdy(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res) {
	float4 d0 = texRead(data_surfobj, c_pix);
	int2 next = c_pix + make_int2(0, 1);
	next.y = clamp(next.y, 0, res.y - 1);
	float4 d1 = texRead(data_surfobj, next);
	return d1 - d0;
}

//updates filtered irradiance and filtered variance
__device__ void psvgf(const IntegratorGlobals& globals, int2 c_pix, int stepSize) {
	// 3x3 kernel from the paper
	const float filterKernel[] =
	{
		0.0625, 0.125, 0.0625,
		0.125, 0.25, 0.125,
		0.0625, 0.125, 0.0625 };

	GBuffer g = sampleGBuffer(globals, c_pix);

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(globals.FrameBuffer.depth_render_surface_object, c_pix, globals.FrameBuffer.resolution).x,
		dFdy(globals.FrameBuffer.depth_render_surface_object, c_pix, globals.FrameBuffer.resolution).x);

	// total irradiance
	float3 irradiance = make_float3(0);
	float variance = 0;

	// weights sum
	float wsum = 0.0;

	//atrous loop
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int2 offset = make_int2(x, y) * stepSize;
			int2 pix = c_pix + offset;
			pix = clamp(pix, make_int2(0, 0), (globals.FrameBuffer.resolution - 1));

			GBuffer s = sampleGBuffer(globals, pix);

			// calculate the normal, depth and luminance weights
			float nw = normalWeight(g.normal, s.normal);
			float dw = depthWeight(g.depth, s.depth, dgrad, make_float2(offset));
			float lw = luminanceWeight(
				getLuminance(RGBSpectrum(g.irradiance)),
				getLuminance(RGBSpectrum(s.irradiance)), g.variance);

			// combine the weights from above
			float w = clamp(nw * dw * lw, 0.f, 1.f);

			// scale by the filtering kernel
			float h = filterKernel[(x + 1) + (y + 1) * 3];
			float hw = w * h;

			// add to total irradiance
			irradiance += s.irradiance * hw;
			variance = s.variance * Sqr(h) * Sqr(w);
			wsum += hw;
		}
	}

	// scale total irradiance by the sum of the weights
	g.irradiance = irradiance / wsum;
	g.variance = variance / Sqr(wsum);
	//write gbuffer
	writeGBuffer(globals, g, c_pix);
}

__global__ void SVGFPass(const IntegratorGlobals globals, int stepsize) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;
	//----------------------------------------------

	int current_objID = texRead(globals.FrameBuffer.objectID_render_surface_object, current_pix).x;

	//void sample/ miss/ sky
	if (current_objID < 0) {
		//no filter
		GBuffer g = sampleGBuffer(globals, current_pix);
		writeGBuffer(globals, g, current_pix);
		return;
	}

	psvgf(globals, current_pix, stepsize);
}

__global__ void composeCompositeImage(const IntegratorGlobals globals) {
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frameres.x, (float)current_pix.y / (float)frameres.y };

	if ((current_pix.x >= frameres.x) || (current_pix.y >= frameres.y)) return;
	//----------------------------------------------

	float4 sampled_irradiance = texRead(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
	RGBSpectrum sampled_radiance = RGBSpectrum(sampled_irradiance);

	float4 sampled_albedo = texRead(globals.FrameBuffer.albedo_render_surface_object, current_pix);

	sampled_radiance *= RGBSpectrum(sampled_albedo);//MODULATE assume BRDF normalised

	RGBSpectrum frag_spectrum = sampled_radiance;
	//EOTF
	frag_spectrum = gammaCorrection(frag_spectrum);
	frag_spectrum = toneMapping(frag_spectrum, 8);

	float4 frag_color = make_float4(frag_spectrum, 1);

	texWrite(frag_color, globals.FrameBuffer.composite_render_surface_object, current_pix);
}

//blits are implemented by renderer; these are proxy
void blitFilteredIrradianceVarianceBackToFront() {};

void blitMomentsBackToFront() {};

//filtered irr back to integrated history back
void blitFilteredIrradianceToHistory() {};

//DONT CALL THIS
void exampleChain(const IntegratorGlobals& globals, dim3 block_dims, dim3 grid_dims) {
	renderPathTraceRaw << < grid_dims, block_dims >> > (globals);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//---
	temporalIntegrate << < grid_dims, block_dims >> > (globals);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	blitMomentsBackToFront();//updates FrontIntegratedHistoryMoments for reads
	//---
	SVGFPass << < grid_dims, block_dims >> > (globals, 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	blitFilteredIrradianceToHistory();//feedback: irradiance from BackFilteredIrradiance
	blitFilteredIrradianceVarianceBackToFront();//updates FrontFiltredIrradiance and FrontIntegratedVariance
	//---
	SVGFPass << < grid_dims, block_dims >> > (globals, 2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	blitFilteredIrradianceVarianceBackToFront();//updates FrontFiltredIrradiance and FrontIntegratedVariance
	//---
	SVGFPass << < grid_dims, block_dims >> > (globals, 4);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	blitFilteredIrradianceVarianceBackToFront();//updates FrontFiltredIrradiance and FrontIntegratedVariance
	//---
	SVGFPass << < grid_dims, block_dims >> > (globals, 8);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	blitFilteredIrradianceVarianceBackToFront();//updates FrontFiltredIrradiance and FrontIntegratedVariance
	//---
	composeCompositeImage << < grid_dims, block_dims >> > (globals);//Display!
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}