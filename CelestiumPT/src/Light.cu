#include "Light.cuh"
#include "ShapeIntersection.cuh"

//#include "Spectrum.cuh"

__device__ LightSampleContext::LightSampleContext(const ShapeIntersection& si)
{
	pos = si.w_pos;
	norm = si.w_geo_norm;
	s_norm = si.w_shading_norm;
}
__device__ LightLiSample Light::SampleLi(LightSampleContext ctx, float2 u2)
{
	ShapeSampleContext shape_ctx{};
	ShapeSample ss = m_triangle->sample(shape_ctx, u2);
	if (ss.pdf == 0 || dot(ss.p - ctx.pos, ss.p - ctx.pos) == 0)return {};
	float3 wi = normalize(ss.p - ctx.pos);
	RGBSpectrum Le = L(ss.p, ss.n, -wi);
	if (!Le)return {};
	return LightLiSample(Le, wi, ss.p, m_triangle->face_normal, ss.pdf);
}