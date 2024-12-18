#pragma once

#include "bounds.cuh"
#include "BLAS.cuh"
#include "TLAS_node.cuh"
#include "bounds.cuh"

#include <cuda_runtime.h>
#include <thrust/universal_vector.h>

#include <vector>

struct IntegratorGlobals;
class Ray;
struct ShapeIntersection;

class TLAS {
public:
	TLAS() = default;
	TLAS(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes);

	int FindBestMatch(int* list, int N, int A, std::vector<TLASNode>& tlasnodes);

	void refresh(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes);

	void build(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes);

	//----------------------------------------------------------------------------------------

	__device__ void intersect(const IntegratorGlobals& globals, const Ray& ray, ShapeIntersection* closest_hitpayload) const;
	__device__ bool intersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax) const;

public:
	Bounds3f m_BoundingBox;
	int m_TLASRootIdx = -1;
	size_t m_TLASnodesCount = 0;
	size_t m_BLASCount = 0;
};