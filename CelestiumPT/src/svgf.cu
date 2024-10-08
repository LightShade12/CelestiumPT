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

__device__ float spatialVarianceEstimate(const IntegratorGlobals& globals, int2 t_current_pix) {
	float3 sampled_normal = normalize(make_float3(texRead(globals.FrameBuffer.world_normals_render_surface_object,
		t_current_pix)));
	float sampled_depth = texRead(globals.FrameBuffer.depth_render_surface_object,
		t_current_pix).x;

	float histlen = texRead(globals.FrameBuffer.current_moments_render_surface_object, t_current_pix).w;

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(globals.FrameBuffer.depth_render_surface_object, t_current_pix, globals.FrameBuffer.resolution).x,
		dFdy(globals.FrameBuffer.depth_render_surface_object, t_current_pix, globals.FrameBuffer.resolution).x);

	float weight_sum = 0.f;
	float2 f_moments = make_float2(0.f);

	int radius = 3; // 7x7 Gaussian Kernel
	for (int yy = -radius; yy <= radius; ++yy)
	{
		for (int xx = -radius; xx <= radius; ++xx)
		{
			int2 offset = make_int2(xx, yy);
			int2 tap_pix = t_current_pix + offset;
			tap_pix = clamp(tap_pix, make_int2(0, 0), (globals.FrameBuffer.resolution - 1));

			float4 tap_irradiance = texRead(globals.FrameBuffer.current_irradiance_render_surface_object,
				tap_pix);
			float3 tap_normal = make_float3(texRead(globals.FrameBuffer.world_normals_render_surface_object,
				tap_pix));
			float tap_depth = texRead(globals.FrameBuffer.depth_render_surface_object,
				tap_pix).x;

			float l = getLuminance(RGBSpectrum(tap_irradiance));

			float nw = normalWeight((sampled_normal), normalize(tap_normal));
			float dw = depthWeight(sampled_depth, tap_depth, dgrad, make_float2(offset));
			float w = clamp(dw * nw, 0.f, 1.f);

			f_moments += make_float2(l, l * l) * w;
			weight_sum += w;
		}
	}

	weight_sum = fmaxf(weight_sum, 1e-6f);
	f_moments /= weight_sum;

	float variance = fabsf(f_moments.y - Sqr(f_moments.x));
	variance *= fminf(4.f / histlen, 1.f);//boost for 1st few frames

	return variance;
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

		texWrite(make_float4(make_float3(0), 0),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		//texWrite(make_float4(final_irradiance, 0),
		//	globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
		//	current_pix);
		//out---
		texWrite(make_float4(final_irradiance, 0),
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
		float var = spatialVarianceEstimate(globals, current_pix);
		//float var = 0;
		texWrite(make_float4(make_float3(var), 0),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		//texWrite(make_float4(final_irradiance, 0),
		//	globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
		//	current_pix);
		//out---
		texWrite(make_float4(final_irradiance, 0),
			globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);

		return;
	}

	//if (prev_px.y < 0)printf(" illegal pixel!");

	bool prj_success = !rejectionHeuristic(globals, prev_px, current_pix);

	//disocclusion/ reproj failure
	if (!prj_success) {
		float var = spatialVarianceEstimate(globals, current_pix);
		//float var = 0;
		texWrite(make_float4(make_float3(var), 0),
			globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix);
		texWrite(make_float4(final_moments.x, final_moments.y, 0, 0),
			globals.FrameBuffer.history_integrated_moments_back_surfobj,
			current_pix);
		//texWrite(make_float4(final_irradiance, 0),
		//	globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
		//	current_pix);
		//out---
		texWrite(make_float4(final_irradiance, 0),
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

	final_moments = lerp(make_float2(hist_moments.x, hist_moments.y), final_moments,
		1.f / fminf(float(moments_hist_len + 1), MAX_ACCUMULATION_FRAMES));

	//feedback: moments
	texWrite(make_float4(final_moments.x, final_moments.y, 0, moments_hist_len + 1),
		globals.FrameBuffer.history_integrated_moments_back_surfobj,
		current_pix);

	//feedback: irradiance
	//texWrite(make_float4(final_irradiance, irradiance_hist_len + 1),
	//	globals.FrameBuffer.history_integrated_irradiance_back_surfobj,
	//	current_pix);

	float variance;
	if (moments_hist_len < 4 || true) {
		variance = spatialVarianceEstimate(globals, current_pix);
	}
	else {
		float2 final_v = final_moments;// / (moments_hist_len);
		variance = fabsf(final_v.y - (Sqr(final_v.x)));
	}

	texWrite(make_float4(make_float3(variance), 1),
		globals.FrameBuffer.filtered_variance_render_front_surfobj,
		current_pix);
	//out----
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

// computes a 3x3 gaussian blur of the variance, centered around
// the current pixel
__device__ float computeVarianceCenter(const IntegratorGlobals& globals, int2 t_current_pix)
{
	float sum = 0.f;

	const float kernel[2][2] = {
		{ 1.0 / 4.0, 1.0 / 8.0  },
		{ 1.0 / 8.0, 1.0 / 16.0 }
	};

	const int radius = 1;
	for (int yy = -radius; yy <= radius; yy++)
	{
		for (int xx = -radius; xx <= radius; xx++)
		{
			int2 tap_pix = t_current_pix + make_int2(xx, yy);
			tap_pix = clamp(tap_pix, make_int2(0, 0), (globals.FrameBuffer.resolution - 1));

			float k = kernel[abs(xx)][abs(yy)];

			sum += texRead(globals.FrameBuffer.filtered_variance_render_front_surfobj,
				tap_pix).x * k;
		}
	}

	return sum;
}

//updates filtered irradiance and filtered variance
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
		float sampled_variance = texRead(globals.FrameBuffer.filtered_variance_render_front_surfobj,
			current_pix).x;
		float4 sampled_irradiance = texRead(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
			current_pix);
		//out----
		texWrite(sampled_irradiance,
			globals.FrameBuffer.filtered_irradiance_back_render_surface_object,
			current_pix);
		texWrite(make_float4(make_float3(sampled_variance), 1),
			globals.FrameBuffer.filtered_variance_render_back_surfobj,
			current_pix);
		return;
	}

	float4 sampled_irradiance = texRead(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
		current_pix);
	float3 sampled_normal = normalize(make_float3(texRead(globals.FrameBuffer.world_normals_render_surface_object,
		current_pix)));
	float sampled_depth = texRead(globals.FrameBuffer.depth_render_surface_object,
		current_pix).x;
	//float sampled_variance = texRead(globals.FrameBuffer.filtered_variance_render_front_surfobj,
	//	current_pix).x;
	float sampled_variance = computeVarianceCenter(globals, current_pix);

	// depth-gradient estimation from screen-space derivatives
	float2 dgrad = make_float2(
		dFdx(globals.FrameBuffer.depth_render_surface_object, current_pix, globals.FrameBuffer.resolution).x,
		dFdy(globals.FrameBuffer.depth_render_surface_object, current_pix, globals.FrameBuffer.resolution).x);

	const float filterKernel[] =
	{
		0.0625, 0.125, 0.0625,
		0.125, 0.25, 0.125,
		0.0625, 0.125, 0.0625 };

	float4 avg_irradiance = make_float4(0);
	float f_variance = 0;
	// weights sum
	float wsum = 0.0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int2 offset = make_int2(x, y) * stepsize;
			int2 tap_pix = current_pix + offset;
			tap_pix = clamp(tap_pix, make_int2(0, 0), (globals.FrameBuffer.resolution - 1));

			float4 tap_irradiance = texRead(globals.FrameBuffer.filtered_irradiance_front_render_surface_object,
				tap_pix);
			float3 tap_normal = make_float3(texRead(globals.FrameBuffer.world_normals_render_surface_object,
				tap_pix));
			float tap_depth = texRead(globals.FrameBuffer.depth_render_surface_object,
				tap_pix).x;
			float tap_variance = texRead(globals.FrameBuffer.filtered_variance_render_front_surfobj,
				tap_pix).x;

			float nw = normalWeight((sampled_normal), normalize(tap_normal));
			float dw = depthWeight(sampled_depth, tap_depth, dgrad, make_float2(offset));
			float lw = 1;
			lw = luminanceWeight(
				getLuminance(RGBSpectrum(sampled_irradiance)),
				getLuminance(RGBSpectrum(tap_irradiance)), sampled_variance);

			// combine the weights from above
			float w = clamp(dw * nw * lw, 0.f, 1.f);

			// scale by the filtering kernel
			float h = filterKernel[(x + 1) + (y + 1) * 3];
			float hw = h * w;

			// add to total irradiance
			avg_irradiance += tap_irradiance * hw;
			f_variance += Sqr(hw) * tap_variance;
			wsum += hw;
		}
	}

	avg_irradiance /= wsum;
	f_variance /= Sqr(wsum);

	avg_irradiance.w = sampled_irradiance.w;//restore history length for temporal feedback

	//out----
	texWrite((avg_irradiance),
		globals.FrameBuffer.filtered_irradiance_back_render_surface_object,
		current_pix);
	texWrite(make_float4(make_float3(f_variance), 1),
		globals.FrameBuffer.filtered_variance_render_back_surfobj,
		current_pix);
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