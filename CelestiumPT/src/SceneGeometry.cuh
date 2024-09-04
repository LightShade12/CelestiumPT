#pragma once
#include "ShapeIntersection.cuh"
class Mesh;
class Ray;
struct Triangle;
struct BVHNode;

//Raw Buffer Data
struct SceneGeometry {
	__device__ ShapeIntersection intersect(const Ray& ray) { return ShapeIntersection(); };
	__device__ bool intersectP(const Ray& ray) { return false; };

	int* DeviceBVHTriangleIndicesBuffer = nullptr;
	size_t DeviceBVHTriangleIndicesCount = 0;

	BVHNode* DeviceBVHNodesBuffer = nullptr;
	size_t DeviceBVHNodesCount = 0;

	Mesh* DeviceMeshesBuffer = nullptr;
	size_t DeviceMeshesCount = 0;

	Triangle* DeviceTrianglesBuffer = nullptr;
	size_t DeviceTrianglesCount = 0;
};