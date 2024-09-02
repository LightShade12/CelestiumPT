#pragma once
#include "vector_types_extension.cuh"

__device__ __host__ bool operator !(const float3& vec);

__device__ float AbsDot(const float3& a, const float3& b);

__device__ bool refract(const float3& wi, float3 normal, float ior, float3& wt);