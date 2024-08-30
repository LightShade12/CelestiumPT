#pragma once

#include "CelestiumPT.hpp"

#include <cstdint>

struct EditorData {
	HostCamera* m_selected_camera = nullptr;//non-owning
	float camera_x_rot_rad = 0;
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
	EditorData m_EditorData;
	GLFWwindow* m_MainWindow = nullptr;
	uint32_t m_width, m_height = 0;
	Renderer m_Renderer;
};