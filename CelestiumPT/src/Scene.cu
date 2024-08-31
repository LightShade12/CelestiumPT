#include "HostScene.hpp"
#include "DeviceScene.cuh"

#include "Triangle.cuh"

#include "Ray.cuh"
#include "ShapeIntersection.cuh"
#include "SceneGeometry.cuh"

HostScene::HostScene(DeviceScene* device_scene)
{
	m_DeviceScene = device_scene;
}

void HostScene::syncDeviceGeometry()
{
	m_DeviceScene->syncDeviceGeometry();
}

void HostScene::AddTriangle(const Triangle& triangle)
{
	Triangle tri = triangle;
	m_DeviceScene->DeviceTriangles.push_back(tri);
	m_DeviceScene->syncDeviceGeometry();
}