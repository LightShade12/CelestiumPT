#pragma once

#include <glm/glm.hpp>

class DeviceMesh;
class HostScene;

class HostMesh {
public:
	HostMesh() = default;
	explicit HostMesh(DeviceMesh* device_mesh);

	void updateDevice(HostScene* hscene);

	glm::mat4 getTransform() const {
		return modelMatrix;
	};

	void setTransform(glm::mat4 transform) {
		modelMatrix = (transform);
	};
public:
	DeviceMesh* m_deviceMesh = nullptr;
	//TODO: guarded access to sensitive data modification
	int triangle_offset_idx = 1;
	size_t tri_count = 0;
	glm::mat4 modelMatrix{};
};