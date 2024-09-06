#pragma once

#include "Mesh.hpp"
#include "Camera.hpp"
#include "CelestiumPT.hpp"

#include <cstdint>

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
	enum class RenderView {
		COMPOSITE = 0,
		NORMALS = 1,
		POSITIONS = 2
	};
	RenderView curent_renderview = RenderView::COMPOSITE;
	Camera m_Camera;
	GLFWwindow* m_MainWindow = nullptr;
	uint32_t m_width, m_height = 0;
	Renderer m_Renderer;
	Mesh m_selected_mesh;
};