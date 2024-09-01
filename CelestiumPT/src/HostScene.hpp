#pragma once
#include <glm/glm.hpp>

class DeviceScene;

class HostScene {
public:
	HostScene() = default;
	explicit HostScene(DeviceScene* device_scene);
	void syncDeviceGeometry();

	size_t getTrianglesCount();

	void AddTriangle(glm::vec3 v0p, glm::vec3 v0n,
		glm::vec3 v1p, glm::vec3 v1n,
		glm::vec3 v2p, glm::vec3 v2n,
		glm::vec3 f_nrm);

private:
	DeviceScene* m_DeviceScene = nullptr;//non-owning; provided initially by renderer
};