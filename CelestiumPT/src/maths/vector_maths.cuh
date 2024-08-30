#pragma once
#include "vector_types_extension.cuh"

__device__ bool refract(const float3& wi, float3 normal, float ior, float3& wt);