#pragma once
#include "Triangle.cuh"
#include "maths/matrix.cuh"
#include "Spectrum.cuh"

struct ShapeIntersection;

struct LightSampleContext {
	__device__ LightSampleContext(const ShapeIntersection& si);
	float3 pos;
	float3 norm;
	float3 s_norm;
};

struct LightLiSample {
	LightLiSample() = default;
	__device__ LightLiSample(const RGBSpectrum& L, const float3& wi, const float3& pLight, const float3& n, float pdf)
		: L(L), wi(wi), pLight(pLight), n(n), pdf(pdf) {}

	RGBSpectrum L;
	float3 wi{};
	float pdf = 0;
	float3 pLight{};//TODO: maybe consider SurfaceInteraction struct
	float3 n;//i think geo
};

class Light {
public:

	__host__ __device__ Light(Triangle* triangle, /*Mat4 transform,*/ float3 color, float power) :
		m_triangle(triangle), /*m_transform(transform),*/ Lemit(color), scale(power)
	{
		area = triangle->area();
	};

	__device__ RGBSpectrum PhiPower() { return make_float3(0); }
	__device__ RGBSpectrum L(float3 p, float3 n, float3 w) const { return scale * Lemit; };
	__device__ LightLiSample SampleLi(LightSampleContext ctx, float2 u2);
	__device__ float PDF_Li(LightSampleContext ctx, float3 wi) { return 0; };

	//Mat4 m_transform;
	RGBSpectrum Lemit;
	float scale;
	Triangle* m_triangle;
	float area;
};