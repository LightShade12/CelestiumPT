#pragma once
#include <vector_types.h>
#include "Ray.cuh"

__constant__ const float HIT_EPSILON = 0.001;

struct ShapeIntersection {
	float hit_distance = -1;
	int triangle_idx = -1;
	float3 bary{};
	float3 w_pos{};
	float3 w_norm{};
	bool front_face = true;

	Ray spawnRay(float3 wi);
};

Ray ShapeIntersection::spawnRay(float3 wi)
{
	float3 orig = w_pos + w_norm * HIT_EPSILON;
	return Ray(orig, wi);
}