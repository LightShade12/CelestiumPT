#pragma once
#include "CelestiumPT.hpp"
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera {
	Camera() {
		position = glm::vec3(0);
		right = { 1,0,0 };
		up = { 0,1,0 };
		forward = { 0,0,-1 };
	};

	Camera(HostCamera* hostcamera) {
		host_camera_handle = hostcamera;
		glm::mat4 mat = host_camera_handle->getTransform();
		position = mat[3];
		forward = mat[2];
		up = mat[1];
		right = mat[0];
		fovYrad = host_camera_handle->FOV_y_radians;
	}

	void resizeFrame(int width, int height) {
		if (width == m_width && height == m_height)return;
		m_width = width; m_height = height;
		recalculateProjection();
		host_camera_handle->updateDevice();
	};

	void recalculateProjection() {
		//printf("Recalculaed proejction\n");
		host_camera_handle->setProjection(glm::perspectiveFovLH(fovYrad, float(m_width), float(m_height), 1.f, 100.f));
		glm::mat4 invmat = glm::inverse(host_camera_handle->getProjection());
		host_camera_handle->setInvProjection(invmat);

		//print_matrix(host_camera_handle->getProjection());
		//print_matrix(host_camera_handle->getInvProjection());
	}
public:
	int m_width = 0, m_height = 0;
	float fovYrad = 0;
	HostCamera* host_camera_handle = nullptr;//non-owning
	float movement_speed = 4;
	float rot_speed = 0.8;

	glm::vec3 position;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 forward;
};