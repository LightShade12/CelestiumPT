#pragma once

#include "glm/glm.hpp"
#include "glm/mat4x4.hpp"

class DeviceCamera;
//source must be compiled by nvcc
class HostCamera {
public:
	HostCamera() = default;
	explicit HostCamera(DeviceCamera* device_camera);
	//apply/construct matrix and send off to device
	void updateDevice();

	void updateCamera();//TODO: proper per frame syncing

	glm::mat4 getTransform() const {
		return m_transform;
	}
	glm::mat4 getView() const {
		return m_view;
	}

	glm::mat4 getProjection() const {
		return m_projection;
	}

	glm::mat4 getInvProjection() const {
		return m_invProjection;
	}

	void setTransform(glm::mat4 mat) {
		m_transform = mat;
		m_view = glm::inverse(m_transform);
	}

	void setProjection(glm::mat4 mat) {
		m_projection = mat;
		m_invProjection = glm::inverse(m_projection);
	}

	float fov_y_radians = glm::radians(60.f);
	float exposure = 4;//set default here
	glm::mat4 m_transform;
	glm::mat4 m_view;
	glm::mat4 m_projection;
	glm::mat4 m_invProjection;
	DeviceCamera* m_device_camera = nullptr;//non-owning link
};