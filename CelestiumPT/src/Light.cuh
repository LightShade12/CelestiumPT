#pragma once
#include "Triangle.cuh"
#include "maths/matrix.cuh"
#include "Spectrum.cuh"

//class RGBSpectrum;

struct LightSampleContext {
	float3 pos;
	float3 norm;
	float3 s_norm;
};

struct LightSample;

class Light {
public:

	__host__ __device__ Light(Triangle* triangle, /*Mat4 transform,*/ float3 color, float power) :
		m_triangle(triangle), /*m_transform(transform),*/ Lemit(color), scale(power)
	{
		float3 edge0 = m_triangle->vertex1.position - m_triangle->vertex0.position;
		float3 edge1 = m_triangle->vertex2.position - m_triangle->vertex0.position;
		area = 0.5f * length(cross(edge0, edge1));
	};

	__device__ RGBSpectrum PhiPower() { return make_float3(0); }
	__device__ RGBSpectrum L(float3 p, float3 n, float3 w) const { return scale * Lemit; };
	__device__ LightSample SampleLi(LightSampleContext ctx);//take uint32_t& seed
	__device__ float PDF_Li(LightSampleContext ctx, float3 wi) { return 0; };

	//Mat4 m_transform;
	RGBSpectrum Lemit;
	float scale;
	Triangle* m_triangle;
	float area;
};