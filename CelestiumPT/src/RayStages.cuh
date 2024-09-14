#pragma once

#include "maths/matrix.cuh"
#include <cuda_runtime.h>

struct ShapeIntersection;
struct CompactShapeIntersection;
struct IntegratorGlobals;
class Ray;
struct Triangle;

//return -1 hit_dist
__device__ ShapeIntersection MissStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload);

__device__ ShapeIntersection ClosestHitStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload);

__device__ void IntersectionStage(const Ray& ray, const Triangle& triangle, int triangle_idx, CompactShapeIntersection* payload);