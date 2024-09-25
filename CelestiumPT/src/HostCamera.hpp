#pragma once

#include "glm/glm.hpp"
#include "glm/mat4x4.hpp"
//#include "glm/gtc/matrix_transform.hpp"

class DeviceCamera;
//source must be compiled by nvcc
class HostCamera {
public:
	HostCamera() = default;
	explicit HostCamera(DeviceCamera* device_camera);
	//apply/construct matrix and send off to device
	void updateDevice();

	glm::mat4 getTransform() const {
		return m_transform;
	}
	glm::mat4 getView() const {
		return m_view;
	}

	void setTransform(glm::mat4 mat) {
		m_transform = mat;
	}

	void setView(glm::mat4 mat) {
		m_view = mat;
	}

	glm::mat4 getProjection() const {
		return m_projection;
	}
	glm::mat4 getInvProjection() const {
		return m_invProjection;
	}

	void setProjection(glm::mat4 mat) {
		m_projection = mat;
	}
	void setInvProjection(glm::mat4 mat) {
		m_invProjection = mat;
	}

	float FOV_y_radians = glm::radians(60.f);
	glm::mat4 m_transform;
	glm::mat4 m_view;
	glm::mat4 m_projection;
	glm::mat4 m_invProjection;
	DeviceCamera* m_device_camera = nullptr;//non-owning link
};