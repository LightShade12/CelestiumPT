#pragma once

#include "CelestiumPT.hpp"

#include <cstdint>

struct EditorData {
	HostCamera* host_camera_handle = nullptr;//non-owning
	float camera_x_rot_rad = 0;
	float camera_y_rot_rad = 0;
	glm::vec3 camera_translation = { 0,0,0 };
};

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
	float speed = 15;
	float rot_speed = 0.8;
	glm::vec3 position;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 forward;
};

struct GLFWwindow;
//TODO: add spdlog; app spec?; Unit tests
class Application
{
public:

	Application();

	void run();

	~Application();

private:
	void initialize();

	void close();

private:
	Camera m_Camera;
	GLFWwindow* m_MainWindow = nullptr;
	uint32_t m_width, m_height = 0;
	Renderer m_Renderer;
};