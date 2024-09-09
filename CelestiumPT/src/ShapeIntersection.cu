#include "ShapeIntersection.cuh"
#include "BSDF.cuh"
#include "Ray.cuh"

__device__ BSDF ShapeIntersection::getBSDF()
{
	return BSDF();
}

__device__ float3 ShapeIntersection::Le()
{
	return make_float3(0);
}

__device__ Ray ShapeIntersection::spawnRay(const float3& wi)
{
	float3 orig = w_pos + (w_norm * HIT_EPSILON);
	return Ray(orig, wi);
}