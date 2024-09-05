#include <glm/glm.hpp>

class Mesh;

class HostMesh {
public:
	HostMesh() = default;
	explicit HostMesh(Mesh* device_mesh);

	void updateDevice();

	glm::mat4 getTransform() const {
		return modelMatrix;
	};

	void setTransform(glm::mat4 transform) {
		modelMatrix = (transform);
	};
public:
	Mesh* m_deviceMesh = nullptr;
	//TODO: guarded access to sensitive data modification
	int triangle_offset_idx = 1;
	size_t tri_count = 0;
	glm::mat4 modelMatrix{};
};