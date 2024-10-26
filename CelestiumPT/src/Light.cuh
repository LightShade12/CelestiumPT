#pragma once
#include "triangle.cuh"
#include "spectrum.cuh"

struct ShapeIntersection;
struct IntegratorGlobals;
class Ray;

struct LightSampleContext {
	__device__ LightSampleContext() = default;
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

class InfiniteLight {
public:
	__host__ __device__ InfiniteLight() { /*Lemit = {0.4,0.7,1.f};*/ scale = 1.f; Lemit = RGBSpectrum(1.f); };
	__host__ __device__ InfiniteLight(float3 color, float power) : Lemit(color), scale(power) {};
	__device__ RGBSpectrum Le(const Ray& ray) const;
	RGBSpectrum Lemit;
	float scale = 1;
};

class Light {
public:

	__host__ __device__ Light(Triangle* triangle, int t_triID, int t_object_id, float3 color, float power) :
		object_id(t_object_id), Lemit(color), scale(power), tri_id(t_triID)
	{
		area = triangle->area();
	};

	__device__ RGBSpectrum PhiPower() const;
	__device__ RGBSpectrum L(float3 p, float3 n, float3 w) const { return scale * Lemit; };
	__device__ LightLiSample SampleLi(const IntegratorGlobals& t_globals, LightSampleContext ctx, float2 u2) const;
	__device__ float PDF_Li(LightSampleContext ctx, float3 wi) const;//TODO:call tri pdf

private:
	int object_id = -1;
	int tri_id = -1;
	RGBSpectrum Lemit;
	float scale;
	float area;
};