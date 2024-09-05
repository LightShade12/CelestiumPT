#pragma once
#include "maths/maths_linear_algebra.cuh"

class Mesh {
public:
	//__device__ ShapeIntersection intersect(const Ray& ray);
	//__device__ bool intersectP(const Ray& ray);

	int triangle_offset_idx = -1;
	size_t tri_count = 0;
	Mat4 modelMatrix;
};