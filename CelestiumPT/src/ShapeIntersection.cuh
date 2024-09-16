#pragma once

//#include "Ray.cuh"
//#include "BSDF.cuh"
//#include "maths/constants.cuh"
#include "maths/maths_linear_algebra.cuh"
#include <cuda_runtime.h>

class BSDF;
class Ray;

struct ShapeIntersection {
	float hit_distance = -1;
	int triangle_idx = -1;
	float3 bary{};
	float3 w_pos{};
	float3 w_norm{};
	//float3 w_shading_norm{};
	//Material material;
	//Light arealight;
	Mat4 invModelMatrix;

	bool front_face = true;
	float3 GAS_debug = make_float3(0);

	__device__ inline bool hasHit() { return triangle_idx != 1; };
	__device__ BSDF getBSDF();
	__device__ float3 Le();

	__device__ Ray spawnRay(const float3& wi);
};

struct CompactShapeIntersection {
	float hit_distance = -1;
	int triangle_idx = -1;
	float3 bary{};
};