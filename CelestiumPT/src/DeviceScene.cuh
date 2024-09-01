#pragma once

#include "Triangle.cuh"
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
	thrust::device_vector<Triangle>DeviceTriangles;
	SceneGeometry* DeviceSceneGeometry = nullptr;

	void syncDeviceGeometry() {
		if (DeviceSceneGeometry == nullptr)return;

		DeviceSceneGeometry->DeviceTrianglesBuffer = thrust::raw_pointer_cast(DeviceTriangles.data());
		DeviceSceneGeometry->DeviceTrianglesCount = DeviceTriangles.size();
	};
};