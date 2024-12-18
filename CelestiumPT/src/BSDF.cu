#include "bsdf.cuh"
#include "device_material.cuh"
#include "samplers.cuh"

BSDF::BSDF(const Mat3& tangent_matrix, RGBSpectrum albedo)
{
	tangentMatrix = tangent_matrix;
	albedo_factor = albedo;
}

__device__ RGBSpectrum BSDF::f(float3 r_wo, float3 r_wi, bool primary_surface) const
{
	//float3 wo = tangentMatrix.inverse() * r_wo;
	float3 wi = tangentMatrix.inverse() * r_wi;
	float3 wo = r_wo;
	//float3 wi = r_wi;
	if (primary_surface)return RGBSpectrum(1.f);

	return fOpaqueDielectric(wo, wi);
}

__device__ float BSDF::pdf(float3 r_wo, float3 r_wi) const
{
	float3 wi = tangentMatrix.inverse() * r_wi;
	float3 wo = r_wo;
	return pdfOpaqueDielectric(wo, wi);
}

__device__ BSDFSample BSDF::sampleBSDF(float3 r_wo, float2 u2, bool primary_surface) const
{
	float3 wo = tangentMatrix.inverse() * r_wo;
	BSDFSample bs = sampleOpaqueDielectric(wo, u2);
	if (primary_surface)bs.f = RGBSpectrum(1.f);
	bs.wi = tangentMatrix * bs.wi;
	return bs;
}

__device__ BSDFSample BSDF::sampleOpaqueDielectric(float3 wo, float2 u2) const
{
	float3 wi = Samplers::sampleCosineWeightedHemisphere(u2);
	RGBSpectrum f = fOpaqueDielectric(wo, wi);
	float pdf = pdfOpaqueDielectric(wo, wi);
	return BSDFSample(f, wi, pdf);
}

__device__ RGBSpectrum BSDF::fOpaqueDielectric(float3 wo, float3 wi) const
{
	return (albedo_factor / PI);
	//return (RGBSpectrum(0.8) / PI);
}

__device__ float BSDF::pdfOpaqueDielectric(float3 wo, float3 wi) const
{
	return AbsDot({ 0,0,1 }, wi) / PI;
}