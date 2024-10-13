#pragma once

#include "maths/linear_algebra.cuh"
#include <cuda_runtime.h>

class RGBSpectrum;
class BSDF;
class Ray;
struct IntegratorGlobals;
class Light;

struct ShapeIntersection {
	float hit_distance = -1;
	int triangle_idx = -1;
	long long int object_idx = -1;
	float3 bary{};
	float2 uv{};
	float3 w_pos{};
	float3 l_pos{};
	float3 w_geo_norm{};

	float3 w_shading_norm{};
	float3 l_shading_norm{};
	//Material material;
	const Light* arealight = nullptr;
	Mat4 m_invModelMatrix;

	bool front_face = true;
	float3 GAS_debug = make_float3(0);

public:
	__device__ inline bool hasHit() const { return triangle_idx != 1; };
	__device__ BSDF getBSDF(const IntegratorGlobals& globals);
	__device__ RGBSpectrum Le(float3 w);//actually wo?

	__device__ Ray spawnRay(const float3& wi) const;
	__device__ Ray spawnRayTo(const float3& p) const;
};

struct CompactShapeIntersection {
	float hit_distance = -1;
	long int triangle_idx = -1;
	float3 bary{};
};