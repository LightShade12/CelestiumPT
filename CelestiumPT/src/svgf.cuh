#pragma once
#include "Spectrum.cuh"
#include "Storage.cuh"
#include "maths/maths_linear_algebra.cuh"

#include <cuda_runtime.h>

__device__ float getLuminance(RGBSpectrum col) {
	// Rec. 709 luminance coefficients for linear RGB
	return 0.2126f * col.r + 0.7152f * col.g + 0.0722f * col.b;
}

//gets raw irradiance
__device__ void integrateMomentsTemporal(RGBSpectrum color) {
	float luminance = getLuminance(color);
	//compute sample moments
	float s = luminance, s_sq = Sqr(luminance);

	//read history moments
	//get history length
	//mix via hist_len
	//write to historyMoments; feedback
}

__device__ void svgf() {
	//read integrated moments
	float4 sampled_moments{};

	float s = sampled_moments.x, s_sq = sampled_moments.y;
	float variance = abs(Sqr(s) - s_sq);//variance must be computed from integrated moments

	//psvgf
}
__global__ void SVGFPass(const IntegratorGlobals& globals, int stepsize) {
	svgf();
	//compute new moments and variance
	//write and update variance
}

/*
* need:
* current_frame_irradiance_buffer
* current_frame_moments_buffer
* current_frame_variance_buffer
* 
* history_integrated_irradiance_buffer
* history_integrated_moments_buffer
*/
void invokeSVGF(dim3 block_dims, dim3 grid_dims, const IntegratorGlobals& globals) {
	//renderKernel must write to irradiance & moments & gbuffer data
	//TAA(irr & moments)(feedback only moments)(then use integrated moments and write to variance 1st iter)
	SVGFPass << < grid_dims, block_dims >> > (globals, 1);//must update variance
	//feedback irradiance
	SVGFPass << < grid_dims, block_dims >> > (globals, 2);
	SVGFPass << < grid_dims, block_dims >> > (globals, 4);
	SVGFPass << < grid_dims, block_dims >> > (globals, 8);
	//Finish! now display
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

struct GBuffer {
	float3 irradiance;
	float3 normal;
	float depth;
	float variance;
};

__device__ GBuffer sampleGBuffer(const IntegratorGlobals& globals, int2 c_pix) {
}

__device__ void writeGBuffer(const IntegratorGlobals& globals, const GBuffer& gbuffer, int2 c_pix) {
}

__device__ float dFdx(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 resolution) {
}

__device__ float dFdy(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 resolution) {
}

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
		dFdx(globals.FrameBuffer.depth_render_surface_object, c_pix, globals.FrameBuffer.resolution),
		dFdy(globals.FrameBuffer.depth_render_surface_object, c_pix, globals.FrameBuffer.resolution));

	// total irradiance
	float3 irradiance = make_float3(0);

	// weights sum
	float wsum = 0.0;

	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			int2 offset = make_int2(x, y) * stepSize;
			GBuffer s = sampleGBuffer(globals, (c_pix + offset));

			// calculate the normal, depth and luminance weights
			float nw = normalWeight(g.normal, s.normal);
			float dw = depthWeight(g.depth, s.depth, dgrad, make_float2(offset));
			float lw = luminanceWeight(
				getLuminance(RGBSpectrum(g.irradiance)),
				getLuminance(RGBSpectrum(s.irradiance)), g.variance);

			// combine the weights from above
			float w = clamp(nw * dw * lw, 0.f, 1.f);

			// scale by the filtering kernel
			w *= filterKernel[(x + 1) + (y + 1) * 3];

			// add to total irradiance
			irradiance += s.irradiance * w;
			wsum += w;
		}
	}

	// scale total irradiance by the sum of the weights
	g.irradiance = irradiance / wsum;

	//write gbuffer
	writeGBuffer(globals, g, c_pix);
}