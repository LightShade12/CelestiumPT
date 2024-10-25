#include "vector_maths.cuh"

__device__ __host__ bool operator!(const float3& vec)
{
	return (vec.x == 0.f && vec.y == 0.f && vec.z == 0.f);
}

__device__ float AbsDot(const float3& a, const float3& b)
{
	return fabsf(dot(a, b));
}

__device__ bool refract(const float3& wi, float3 normal, float ior, float3& wt)
{
	float cosTheta = dot(wi, normal);

	if (cosTheta < 0.0f) {
		ior = 1.0f / ior;
		cosTheta *= -1.0f;
		normal *= -1.0f;
	}

	float sin2Theta = (1.0f - cosTheta * cosTheta);
	float sin2Theta_t = sin2Theta / (ior * ior);
	if (sin2Theta_t >= 1.0f) return false;

	float cosTheta_t = sqrtf(1.0f - sin2Theta_t);
	wt = (-1.f * wi) / ior + (cosTheta / ior - cosTheta_t) * normal;
	return true;
}

__device__ bool checkNaN(const float3& vec)
{
	return isnan(vec.x) || isnan(vec.y) || isnan(vec.z);
}

__device__ bool checkINF(const float3& vec)
{
	return isinf(vec.x) || isinf(vec.y) || isinf(vec.z);
}

__device__ float3 log2f(const float3 a)
{
	return make_float3(log2f(a.x), log2f(a.y), log2f(a.z));
}

__device__ float3 powf(const float3 a, const float3 b)
{
	return make_float3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z));
}
