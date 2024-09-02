#pragma once
#include "CelestiumPT.hpp"
#include <glm/glm.hpp>

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
	}

	HostCamera* host_camera_handle = nullptr;//non-owning
	float movement_speed = 4;
	float rot_speed = 0.8;
	glm::vec3 position;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 forward;
};