#pragma once

//#include "ShapeIntersection.cuh"
#include "acceleration_structure/GAS.cuh"

class DeviceMesh;
class Ray;
struct Triangle;
struct BVHNode;
class BLAS;
class TLASNode;

//Raw Buffer Data
struct SceneGeometry {
	//__device__ ShapeIntersection intersect(const Ray& ray) { return ShapeIntersection(); };
	//__device__ bool intersectP(const Ray& ray) { return false; };

	GAS GAS_structure;

	int* DeviceBVHTriangleIndicesBuffer = nullptr;
	size_t DeviceBVHTriangleIndicesCount = 0;

	BVHNode* DeviceBVHNodesBuffer = nullptr;
	size_t DeviceBVHNodesCount = 0;

	BLAS* DeviceBLASesBuffer = nullptr;
	size_t DeviceBLASesCount = 0;

	TLASNode* DeviceTLASNodesBuffer = nullptr;
	size_t DeviceTLASNodesCount = 0;

	DeviceMesh* DeviceMeshesBuffer = nullptr;
	size_t DeviceMeshesCount = 0;

	Triangle* DeviceTrianglesBuffer = nullptr;
	size_t DeviceTrianglesCount = 0;
};