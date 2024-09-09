#pragma once

//#include "Ray.cuh"
//#include "BSDF.cuh"
//#include "maths/constants.cuh"
#include <cuda_runtime.h>
#include <vector_types.h>

class BSDF;
class Ray;

struct ShapeIntersection {
	float hit_distance = -1;
	int triangle_idx = -1;
	float3 bary{};
	float3 w_pos{};
	float3 w_norm{};
	bool front_face = true;

	__device__ BSDF getBSDF();
	__device__ float3 Le();

	__device__ Ray spawnRay(const float3& wi);
};