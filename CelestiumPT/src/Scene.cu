#include "HostScene.hpp"
#include "DeviceScene.cuh"

#include "Triangle.cuh"

#include "Ray.cuh"
#include "ShapeIntersection.cuh"
//#include "SceneGeometry.cuh"

HostScene::HostScene(DeviceScene* device_scene)
{
	m_DeviceScene = device_scene;
}

void HostScene::syncDeviceGeometry()
{
	m_DeviceScene->syncDeviceGeometry();
}

size_t HostScene::getTrianglesCount() {
	return m_DeviceScene->DeviceTriangles.size();
}

void HostScene::AddTriangle(
	glm::vec3 v0p, glm::vec3 v0n,
	glm::vec3 v1p, glm::vec3 v1n,
	glm::vec3 v2p, glm::vec3 v2n,
	glm::vec3 f_nrm)
{
	Triangle tri(
		Vertex(v0p, v0n),
		Vertex(v1p, v1n),
		Vertex(v2p, v2n),
		f_nrm
	);
	m_DeviceScene->DeviceTriangles.push_back(tri);
	m_DeviceScene->syncDeviceGeometry();
}