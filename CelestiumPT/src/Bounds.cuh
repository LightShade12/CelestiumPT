#pragma once

#include "maths/maths_linear_algebra.cuh"
#include <float.h>
#include <cuda_runtime.h>

class Ray;

struct Bounds3f
{
	Bounds3f() = default;
	Bounds3f(float3 min, float3 max) :pMin(min), pMax(max) {};

	__host__ void adaptBounds(const Mat4& model_mat, const Bounds3f& origin);
	__device__ float intersect(const Ray& ray) const;
	float getSurfaceArea() const;//TODO: will return fltmin, fltmax if uninitialised
	float3 getCentroid() const;

	float3 pMin = { FLT_MAX,FLT_MAX, FLT_MAX };
	float3 pMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
};