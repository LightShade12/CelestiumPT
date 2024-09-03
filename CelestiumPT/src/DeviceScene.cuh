#pragma once

#include "Triangle.cuh"
#include "Mesh.cuh"
#include "SceneGeometry.cuh"
#include <thrust/device_vector.h>

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

		DeviceSceneGeometry->DeviceMeshesBuffer = thrust::raw_pointer_cast(DeviceMeshes.data());
		DeviceSceneGeometry->DeviceMeshesCount = DeviceMeshes.size();
	};

public:

	SceneGeometry* DeviceSceneGeometry = nullptr;
	thrust::device_vector<Triangle>DeviceTriangles;
	thrust::device_vector<Mesh>DeviceMeshes;
};