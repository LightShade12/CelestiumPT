#pragma once

#include "HostMesh.hpp"
#include "HostCamera.hpp"
#include <glm/glm.hpp>

class DeviceScene;

class HostScene {
public:
	HostScene() = default;
	explicit HostScene(DeviceScene* device_scene);
	void syncDeviceGeometry();

	size_t getTrianglesCount();
	size_t getMeshesCount();

	void setCamera(HostCamera camera);

	void AddTriangle(glm::vec3 v0p, glm::vec3 v0n,
		glm::vec3 v1p, glm::vec3 v1n,
		glm::vec3 v2p, glm::vec3 v2n,
		glm::vec3 f_nrm, bool skip_sync = true);

	void AddMesh(HostMesh hmesh);
	void LogStatus();

	HostMesh getMesh(size_t mesh_idx);

	DeviceScene* m_DeviceScene = nullptr;//non-owning; provided initially by renderer
};