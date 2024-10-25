#pragma once
#include "celestium_pt.hpp"
#include "glm/glm.hpp"

class Mesh {
public:
	Mesh() = default;
	explicit Mesh(HostMesh host_mesh) :host_mesh_handle(host_mesh), scale(1), original_tranform(1) {
		original_tranform = host_mesh_handle.getTransform();
	};

	glm::vec3 translation{};
	glm::vec3 scale{};
	glm::vec3 rotation{};

	glm::mat4 original_tranform{};
	HostMesh host_mesh_handle;//non owning
};