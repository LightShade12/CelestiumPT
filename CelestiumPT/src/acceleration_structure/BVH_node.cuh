#pragma once

#include "bounds.cuh"

//BLAS Node
struct BVHNode {
	BVHNode() = default;

	Bounds3f m_BoundingBox;
	long int left_child_or_triangle_indices_start_idx = -1;	//TODO:make them uint?
	int triangle_indices_count = 0;

	float getSurfaceArea() const
	{
		if (triangle_indices_count == 0)
			return 0;

		return m_BoundingBox.getSurfaceArea();
	}
};