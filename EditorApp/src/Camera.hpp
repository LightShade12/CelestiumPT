#pragma once
#include "celestium_pt.hpp"
#include "glm/glm.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

struct Camera {
	Camera() {
		position = glm::vec3(0);
		right = { 1,0,0 };
		up = { 0,1,0 };
		forward = { 0,0,-1 };
	};

	Camera(HostCamera* hostcamera) {
		host_camera_handle = hostcamera;

		//transform
		glm::mat4 mat = host_camera_handle->getTransform();
		position = mat[3];
		forward = mat[2];
		up = mat[1];
		right = mat[0];
		
		//imaging
		exposure = host_camera_handle->exposure;
		fov_y_rad = host_camera_handle->fov_y_radians;
	}

	void resizeFrame(int t_width, int t_height) {
		if (t_width == width && t_height == height)return;
		width = t_width; height = t_height;
		recalculateProjection();
		host_camera_handle->updateDevice();
	};

	void recalculateProjection() {
		host_camera_handle->setProjection(
			glm::perspectiveFovLH(fov_y_rad, float(width), float(height), 1.f, 100.f));
	}
public:
	int width = 0, height = 0;
	float fov_y_rad = 0;
	HostCamera* host_camera_handle = nullptr;//non-owning
	float movement_speed = 6;
	float rot_speed = 0.8;

	float exposure = 1;
	glm::vec3 position;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 forward;
};