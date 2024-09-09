#pragma once

#include "Triangle.cuh"
#include "DeviceMesh.cuh"
#include "acceleration_structure/BVHNode.cuh"
#include "acceleration_structure/BLAS.cuh"
#include "acceleration_structure/TLASNode.cuh"

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

struct SceneGeometry;

class DeviceScene {
public:

	DeviceScene() = default;

	DeviceScene(SceneGeometry* device_scene_geo);

	void syncDeviceGeometry();

public:

	SceneGeometry* DeviceSceneGeometry = nullptr;
	thrust::universal_vector<Triangle>DeviceTriangles;
	thrust::universal_vector<BVHNode>DeviceBVHNodes;
	thrust::universal_vector<BLAS>DeviceBLASes;
	thrust::universal_vector<TLASNode>DeviceTLASNodes;
	thrust::universal_vector<int>DeviceBVHTriangleIndices;
	thrust::universal_vector<DeviceMesh>DeviceMeshes;
};