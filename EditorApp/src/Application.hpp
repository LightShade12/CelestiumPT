#pragma once

#include "EditorSandbox.hpp"

#include <cstdint>

struct GLFWwindow;
//TODO: add spdlog; app spec?; Unit tests
class Application
{
public:

	Application();

	void run();

	GLFWwindow* getWindowHandle() { return m_MainWindow; };
	static Application& Get();

	~Application();

private:
	void initialize();

	void close();

private:

	EditorSandbox m_EditorSandbox;
	GLFWwindow* m_MainWindow = nullptr;
	uint32_t m_width, m_height = 0;
};