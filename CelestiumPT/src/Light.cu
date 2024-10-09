#include "light.cuh"
#include "shape_intersection.cuh"
#include "maths/constants.cuh"
#include "ray.cuh"

__device__ RGBSpectrum SkyShading(const Ray& ray) {
	//return make_float3(0);
	float3 unit_direction = normalize(ray.getDirection());
	float a = 0.5f * (unit_direction.y + 1.0);
	//return make_float3(0.2f, 0.3f, 0.4f);
	return (1.0f - a) * RGBSpectrum(1.0, 1.0, 1.0) + a * RGBSpectrum(0.2, 0.4, 1.0);
};

__device__ RGBSpectrum InfiniteLight::Le(const Ray& ray) const
{
	return SkyShading(ray) * scale * Lemit;
}
__device__ LightSampleContext::LightSampleContext(const ShapeIntersection& si)
{
	pos = si.w_pos;
	norm = si.w_geo_norm;
	s_norm = si.w_shading_norm;
}
__device__ RGBSpectrum Light::PhiPower() const
{
	return scale * Lemit * area * PI * 2.f;
}
__device__ LightLiSample Light::SampleLi(LightSampleContext ctx, float2 u2) const
{
	ShapeSampleContext shape_ctx{};
	ShapeSample ss = m_triangle->sample(shape_ctx, u2);
	if (ss.pdf == 0 || dot(ss.p - ctx.pos, ss.p - ctx.pos) == 0)return {};
	float3 wi = normalize(ss.p - ctx.pos);
	RGBSpectrum Le = L(ss.p, ss.n, -wi);
	if (!Le)return {};
	return LightLiSample(Le, wi, ss.p, m_triangle->face_normal, ss.pdf);
}

__device__ float Light::PDF_Li(LightSampleContext ctx, float3 wi) const
{
	return 1.f / area;
}