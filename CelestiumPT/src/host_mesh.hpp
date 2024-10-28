#pragma once

#include <glm/glm.hpp>
#include <string>

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
	glm::mat4 getInverseTransform() const {
		return m_invModelMatrix;
	};

	void setTransform(glm::mat4 transform) {
		modelMatrix = transform;
		m_invModelMatrix = glm::inverse(modelMatrix);
	};
public:
	std::string name;
	DeviceMesh* m_deviceMesh = nullptr;
	//TODO: guarded access to sensitive m_data modification
	int triangle_offset_idx = 1;
	size_t tri_count = 0;

	glm::mat4 modelMatrix{};
	glm::mat4 m_invModelMatrix{};
};