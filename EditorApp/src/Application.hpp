#pragma once

#include "Camera.hpp"
#include "CelestiumPT.hpp"

#include <cstdint>

struct EditorData {
	HostCamera* host_camera_handle = nullptr;//non-owning
	float camera_x_rot_rad = 0;
	float camera_y_rot_rad = 0;
	glm::vec3 camera_translation = { 0,0,0 };
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