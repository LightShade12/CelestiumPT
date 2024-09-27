#include "maths_linear_algebra.cuh"
#include "constants.cuh"

__host__ __device__ float deg2rad(float degree)
{
	return (degree * (PI / 180.f));
}

__host__ __device__ float Sqr(float v) {
	return v * v;
}