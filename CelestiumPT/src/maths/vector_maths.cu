#include "vector_maths.cuh"

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