#pragma once

#include "maths/linear_algebra.cuh"

class DeviceMesh {
public:
	//__device__ ShapeIntersection intersect(const Ray& ray);
	//__device__ bool intersectP(const Ray& ray);

	char name[32];
	long int BLAS_idx = -1;

	int triangle_offset_idx = -1;
	size_t tri_count = 0;

	Mat4 modelMatrix, prev_modelMatrix;
	Mat4 inverseModelMatrix;
};