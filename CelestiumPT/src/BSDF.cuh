#pragma once

#include "maths/matrix.cuh"
#include "Spectrum.cuh"
#include "maths/constants.cuh"

struct BSDFSample {
	__device__ BSDFSample() = default;
	__device__ BSDFSample(RGBSpectrum f, float3 wi, float pdf) :f(f), wi(wi), pdf(pdf) {};
	RGBSpectrum f{};
	float3 wi{};
	float pdf = 0;
};

class BSDF {
public:

	__device__ BSDF(float3 x, float3 y, float3 z) :tangentMatrix(Mat3(x, y, z)) {};

	//BSDF(float3 tang, float3 bitan, float3 n) :tangentMatrix(Mat3(tang, bitan, n)) {};

	__device__ BSDF(const Mat3& tangent_matrix) :tangentMatrix(tangent_matrix) {};

	__device__ RGBSpectrum f(float3 r_wo, float3 r_wi);

	__device__ float pdf(float3 r_wo, float3 r_wi);

	__device__ BSDFSample sampleBSDF(float3 r_wo, float2 u2);

	__device__ BSDFSample sampleOpaqueDielectric(float3 wo, float2 u2);

	__device__ RGBSpectrum fOpaqueDielectric(float3 wo, float3 wi);

	__device__ float pdfOpaqueDielectric(float3 wo, float3 wi);

public:

	Mat3 tangentMatrix;
};