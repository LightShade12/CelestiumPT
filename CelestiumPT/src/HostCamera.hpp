#pragma once
#include "glm/glm.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

class DeviceCamera;
//source must be compiled by nvcc
class HostCamera {
public:
	HostCamera() = default;
	explicit HostCamera(DeviceCamera* dev_camera);
	//apply/construct matrix and send off to device
	void updateDevice();

	glm::mat4 getTransform() const {
		return m_transform;
	}

	void setTransform(glm::mat4 mat) {
		m_transform = mat;
	}

	float* getTranslationPtr() {
		return &(m_transform[3][0]);
	}

	void rotate(glm::vec3 axis, float radians) {
		m_transform = glm::rotate(m_transform, radians, axis);
	}
	void translate(glm::vec3 pos) {
		m_transform = glm::translate(m_transform, pos);
	}

public:
	glm::mat4 m_transform;
	DeviceCamera* m_device_camera = nullptr;//non-owning link
};