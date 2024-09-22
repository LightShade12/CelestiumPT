#include "BSDF.cuh"
#include "samplers.cuh"

__device__ float3 BSDF::f(float3 r_wo, float3 r_wi)
{
	//float3 wo = tangentMatrix.inverse() * r_wo;
	float3 wi = tangentMatrix.inverse() * r_wi;
	float3 wo = r_wo;
	//float3 wi = r_wi;
	return fOpaqueDielectric(wo, wi);
}

__device__ float BSDF::pdf(float3 r_wo, float3 r_wi)
{
	float3 wi = tangentMatrix.inverse() * r_wi;
	float3 wo = r_wo;
	return pdfOpaqueDielectric(wo, wi);
}

__device__ BSDFSample BSDF::sampleBSDF(float3 r_wo, float2 u2)
{
	float3 wo = tangentMatrix.inverse() * r_wo;
	BSDFSample bs = sampleOpaqueDielectric(wo, u2);
	bs.wi = tangentMatrix * bs.wi;
	return bs;
}

__device__ BSDFSample BSDF::sampleOpaqueDielectric(float3 wo, float2 u2)
{
	float3 wi = Samplers::sampleCosineWeightedHemisphere(u2);
	float3 f = fOpaqueDielectric(wo, wi);
	float pdf = pdfOpaqueDielectric(wo, wi);
	return BSDFSample(f, wi, pdf);
}

__device__ float3 BSDF::fOpaqueDielectric(float3 wo, float3 wi)
{
	return make_float3(0.8f) / PI;
}

__device__ float BSDF::pdfOpaqueDielectric(float3 wo, float3 wi)
{
	return AbsDot({ 0,0,1 }, wi) / PI;
}