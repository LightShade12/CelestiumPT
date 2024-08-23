#pragma once

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
	GLFWwindow* m_MainWindow = nullptr;
	uint32_t m_width, m_height = 0;
	Renderer m_Renderer;
};