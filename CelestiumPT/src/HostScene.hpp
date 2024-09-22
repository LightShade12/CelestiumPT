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
	size_t getCamerasCount();

	void addCamera(HostCamera camera);

	void AddTriangle(
		glm::vec3 v0p, glm::vec3 v0n, glm::vec2 v0uv,
		glm::vec3 v1p, glm::vec3 v1n, glm::vec2 v1uv,
		glm::vec3 v2p, glm::vec3 v2n, glm::vec2 v2uv,
		glm::vec3 f_nrm, bool skip_sync = true);

	void addLight(int triangle_idx, glm::vec3 color, float scale);
	void AddMesh(HostMesh hmesh);
	void LogStatus();

	HostMesh getMesh(size_t mesh_idx);

	DeviceScene* m_DeviceScene = nullptr;//non-owning; provided initially by renderer
};