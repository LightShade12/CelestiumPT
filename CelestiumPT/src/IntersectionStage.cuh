#pragma once
#include "ShapeIntersection.cuh"
#include "Triangle.cuh"
#include <cuda_runtime.h>

__device__ ShapeIntersection IntersectionStage(const Ray& ray, const Triangle& triangle, int triangle_idx);