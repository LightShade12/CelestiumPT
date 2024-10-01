#pragma once
#include "Spectrum.cuh"

struct IntegratorGlobals;
struct ShapeIntersection;

__device__ RGBSpectrum agx_fitted(RGBSpectrum col);
__device__ RGBSpectrum agx_fitted_Eotf(RGBSpectrum col);

__device__ RGBSpectrum agx_tonemapping(RGBSpectrum /*Linear BT.709*/col);

__device__ RGBSpectrum uncharted2_tonemap_partial(RGBSpectrum x);

__device__ RGBSpectrum uncharted2_filmic(RGBSpectrum v, float exposure);

__device__ RGBSpectrum toneMapping(RGBSpectrum HDR_color, float exposure = 2.f);

__device__ RGBSpectrum gammaCorrection(const RGBSpectrum linear_color);

__device__ void recordGBufferHit(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si);

__device__ void recordGBufferAny(const IntegratorGlobals& globals, float2 ppixel, const ShapeIntersection& si);

__device__ void recordGBufferMiss(const IntegratorGlobals& globals, float2 ppixel);

class Film {
public:
	void addSample() {
		//sensor conversion=white balance; tonemap
		//clamp
		//add sample/average
	};
};