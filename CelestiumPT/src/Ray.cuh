#pragma once
#include "maths/maths_linear_algebra.cuh"

class Ray {
public:

	__device__ Ray(float3 orig, float3 dir) :origin(orig), direction(dir), invdirection(1.f / dir) {};

	__device__ void setOrigin(float3 new_orig) { origin = new_orig; };
	__device__ void setDirection(float3 new_dir) { direction = new_dir; invdirection = 1.f / direction; };

	__device__ inline float3 getOrigin() const { return origin; };
	__device__ inline float3 getDirection() const { return direction; };
	__device__ inline float3 getInvDirection() const { return invdirection; };
private:
	float3 origin;
	float3 direction;
	float3 invdirection;
};