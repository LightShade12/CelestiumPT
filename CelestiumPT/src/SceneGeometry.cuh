#pragma once
#include "ShapeIntersection.cuh"
class Mesh;
class Ray;
struct Triangle;

//Raw Buffer Data
struct SceneGeometry {
	__device__ ShapeIntersection intersect(const Ray& ray) { return ShapeIntersection(); };
	__device__ bool intersectP(const Ray& ray) { return false; };

	Mesh* DeviceMeshesBuffer = nullptr;
	size_t DeviceMeshesCount = 0;
	Triangle* DeviceTrianglesBuffer = nullptr;
	size_t DeviceTrianglesCount = 0;
};