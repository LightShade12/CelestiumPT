#pragma once

#include "triangle.cuh"

#include <vector_types.h>
#include <float.h>

struct BVHPrimitiveBounds {
	BVHPrimitiveBounds() = default;
	BVHPrimitiveBounds(float3 min, float3 max) :min(min), max(max) {};

	float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
};

struct BVHPrimitiveCache {
	BVHPrimitiveCache(Triangle triangle);
	float3 centroid;
	BVHPrimitiveBounds bounds;
};

