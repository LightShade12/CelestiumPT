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

size_t HostScene::getMeshesCount() {
	return m_DeviceScene->DeviceMeshes.size();
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

void HostScene::AddMesh(HostMesh hmesh)
{
	Mesh dmesh;
	dmesh.triangle_offset_idx = hmesh.triangle_offset_idx;
	dmesh.tri_count = hmesh.tri_count;
	Mat4 mat(
		hmesh.modelMatrix[0][0], hmesh.modelMatrix[0][1], hmesh.modelMatrix[0][2], hmesh.modelMatrix[0][3],  // First column
		hmesh.modelMatrix[1][0], hmesh.modelMatrix[1][1], hmesh.modelMatrix[1][2], hmesh.modelMatrix[1][3],  // Second column
		hmesh.modelMatrix[2][0], hmesh.modelMatrix[2][1], hmesh.modelMatrix[2][2], hmesh.modelMatrix[2][3],  // Third column
		hmesh.modelMatrix[3][0], hmesh.modelMatrix[3][1], hmesh.modelMatrix[3][2], hmesh.modelMatrix[3][3]   // Fourth column
	);
	dmesh.modelMatrix = mat;
	m_DeviceScene->DeviceMeshes.push_back(dmesh);
	m_DeviceScene->syncDeviceGeometry();
}