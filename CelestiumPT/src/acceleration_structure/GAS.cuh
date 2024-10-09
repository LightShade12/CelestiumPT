#pragma once

#include "BLAS_builder.hpp"
#include "TLAS.cuh"

#include <cuda_runtime.h>

class Ray;
class HostScene;
struct ShapeIntersection;
struct IntegratorGlobals;

class GAS {
public:
	GAS() = default;

	void build(HostScene* host_scene);
	void refresh(HostScene* host_scene);

	__device__ ShapeIntersection intersect(const IntegratorGlobals& globals, const Ray& ray, float tmax = FLT_MAX);
	__device__ bool intersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax);

	TLAS tlas;
	BLASBuilder blasbuilder;
private:
};