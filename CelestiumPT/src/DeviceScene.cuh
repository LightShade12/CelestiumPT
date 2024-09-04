#pragma once

#include "Triangle.cuh"
#include "Mesh.cuh"
#include "acceleration_structure/BVHNode.cuh"
#include "SceneGeometry.cuh"
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

class DeviceScene {
public:

	DeviceScene() = default;

	DeviceScene(SceneGeometry* device_scene_geo) :DeviceSceneGeometry(device_scene_geo)
	{
		if (DeviceSceneGeometry == nullptr) {
			cudaMallocManaged(&DeviceSceneGeometry, sizeof(DeviceSceneGeometry));
		}
		syncDeviceGeometry();
	};

	void syncDeviceGeometry() {
		if (DeviceSceneGeometry == nullptr)return;

		DeviceSceneGeometry->DeviceTrianglesBuffer = thrust::raw_pointer_cast(DeviceTriangles.data());
		DeviceSceneGeometry->DeviceTrianglesCount = DeviceTriangles.size();

		DeviceSceneGeometry->DeviceBVHTriangleIndicesBuffer = thrust::raw_pointer_cast(DeviceBVHTriangleIndices.data());
		DeviceSceneGeometry->DeviceBVHTriangleIndicesCount = DeviceBVHTriangleIndices.size();

		DeviceSceneGeometry->DeviceBVHNodesBuffer = thrust::raw_pointer_cast(DeviceBVHNodes.data());
		DeviceSceneGeometry->DeviceBVHNodesCount = DeviceBVHNodes.size();

		DeviceSceneGeometry->DeviceMeshesBuffer = thrust::raw_pointer_cast(DeviceMeshes.data());
		DeviceSceneGeometry->DeviceMeshesCount = DeviceMeshes.size();
	};

public:

	SceneGeometry* DeviceSceneGeometry = nullptr;
	thrust::universal_vector<Triangle>DeviceTriangles;
	thrust::device_vector<BVHNode>DeviceBVHNodes;
	thrust::device_vector<int>DeviceBVHTriangleIndices;
	thrust::universal_vector<Mesh>DeviceMeshes;
};