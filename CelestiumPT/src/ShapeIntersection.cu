#include "ShapeIntersection.cuh"
#include "BSDF.cuh"
#include "Ray.cuh"
#include "Spectrum.cuh"

__device__ BSDF ShapeIntersection::getBSDF()
{
	return BSDF();
}

__device__ RGBSpectrum ShapeIntersection::Le(float3 w)
{

	//if(arealight!=nullptr)printf("light scale: %.3f | ", arealight->scale);
	return (arealight != nullptr) ?
		arealight->L(w_pos, w_norm, w) :
		make_float3(0);
}

__device__ Ray ShapeIntersection::spawnRay(const float3& wi)
{
	float3 orig = w_pos + (w_norm * HIT_EPSILON);
	return Ray(orig, wi);
}