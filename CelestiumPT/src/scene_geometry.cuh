#pragma once
//TODO:This file is stupid
#include "light.cuh"
#include "acceleration_structure/GAS.cuh"

//class Light;
class DeviceMesh;
class Ray;
struct Triangle;
struct BVHNode;
class BLAS;
class TLASNode;
struct DeviceMaterial;
class DeviceTexture;

//Raw Buffer Data
struct SceneGeometry {
	//__device__ ShapeIntersection intersect(const Ray& ray) { return ShapeIntersection(); };
	//__device__ bool intersectP(const Ray& ray) { return false; };

	GAS GAS_structure;
	InfiniteLight SkyLight;

	Light* DeviceLightsBuffer = nullptr;
	size_t DeviceLightsCount = 0;

	size_t* DeviceBVHTriangleIndicesBuffer = nullptr;
	size_t DeviceBVHTriangleIndicesCount = 0;

	BVHNode* DeviceBVHNodesBuffer = nullptr;
	size_t DeviceBVHNodesCount = 0;

	BLAS* DeviceBLASesBuffer = nullptr;
	size_t DeviceBLASesCount = 0;

	TLASNode* DeviceTLASNodesBuffer = nullptr;
	size_t DeviceTLASNodesCount = 0;

	DeviceMesh* DeviceMeshesBuffer = nullptr;
	size_t DeviceMeshesCount = 0;

	DeviceTexture* DeviceTexturesBuffer = nullptr;
	size_t DeviceTexturesCount = 0;

	DeviceMaterial* DeviceMaterialBuffer = nullptr;
	size_t DeviceMaterialsCount = 0;

	Triangle* DeviceTrianglesBuffer = nullptr;
	size_t DeviceTrianglesCount = 0;
};