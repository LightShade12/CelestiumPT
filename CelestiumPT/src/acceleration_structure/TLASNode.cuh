#pragma once

#include "Bounds.cuh"
#include <cstdint>

class TLASNode {
public:

	TLASNode() = default;

	Bounds3f m_BoundingBox;
	uint32_t leftRight = 0;
	int BLAS_idx = 0;

	bool isleaf() const {
		return (leftRight == 0);
	};

	//TODO: how to handle this part
	float getSurfaceArea() const
	{
		if (!isleaf())
			return m_BoundingBox.getSurfaceArea();

		return m_BoundingBox.getSurfaceArea();
	}
};