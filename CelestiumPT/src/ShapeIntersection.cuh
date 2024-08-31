#pragma once
#include <vector_types.h>
	
struct ShapeIntersection {
	float hit_distance = -1;
	int triangle_idx = -1;
	float3 bary{};
	float3 w_pos{};
	float3 w_norm{};
	bool front_face = true;
};