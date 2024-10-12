#pragma once

#include "triangle.cuh"
#include "device_mesh.cuh"
#include "device_material.cuh"
#include "device_camera.cuh"
#include "Light.cuh"
#include "acceleration_structure/BVH_node.cuh"
#include "acceleration_structure/BLAS.cuh"
#include "acceleration_structure/TLAS_node.cuh"

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
	thrust::universal_vector<Light>DeviceLights;
	thrust::universal_vector<BVHNode>DeviceBVHNodes;
	thrust::universal_vector<BLAS>DeviceBLASes;
	thrust::universal_vector<TLASNode>DeviceTLASNodes;
	thrust::universal_vector<size_t>DeviceBVHTriangleIndices;
	thrust::universal_vector<DeviceMesh>DeviceMeshes;
	thrust::universal_vector<DeviceMaterial>DeviceMaterials;
	thrust::universal_vector<DeviceCamera>DeviceCameras;
};