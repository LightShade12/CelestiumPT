#include "Application.hpp"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include <glad/include/glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <glfw/include/GLFW/glfw3.h>
#include <iostream>

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

Application::Application()
{
	initialize();
	m_EditorData.m_selected_camera = m_Renderer.getCurrentCamera();
}

Application::~Application()
{
	close();
}

void Application::run()
{
	fprintf(stdout, "start run..\n");
	//int width, height;
	glClearColor(0.f, 0.24f, 0.3f, 1.f);

	while (!glfwWindowShouldClose(m_MainWindow))
	{
		glfwPollEvents();

		if (glfwGetWindowAttrib(m_MainWindow, GLFW_ICONIFIED) != 0)
		{
			ImGui_ImplGlfw_Sleep(10);
			continue;
		}

		glClear(GL_COLOR_BUFFER_BIT);
		glfwGetFramebufferSize(m_MainWindow, (int*)&m_width, (int*)&m_height);
		glViewport(0, 0, m_width, m_height);

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		{
			ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

			ImGui::Begin("Hello World");
			ImGui::Text("This is a window");
			bool updateCam = false;

			updateCam |= ImGui::InputFloat3("Camera translation", m_EditorData.m_selected_camera->getTranslationPtr());
			if (ImGui::SliderAngle("camera X axis rot", &m_EditorData.camera_x_rot_rad))
			{
				updateCam |= true; m_EditorData.m_selected_camera->rotate({ 1,0,0 }, 0.5);
			};

			if (updateCam)m_EditorData.m_selected_camera->updateDevice();
			ImGui::End();

			ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);
			ImVec2 vpdims = ImGui::GetContentRegionAvail();
			if (m_Renderer.getCompositeRenderTargetTextureName() != NULL)
				ImGui::Image((void*)(uintptr_t)m_Renderer.getCompositeRenderTargetTextureName(),
					ImVec2((float)m_Renderer.getFrameWidth(), (float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			ImGui::End();

			if (vpdims.y > 14)vpdims.y -= 12;//TODO: make this sensible var; not a constant
			if (vpdims.y < 5)vpdims.y = 10;
			m_Renderer.resizeResolution((int)vpdims.x, (int)vpdims.y);
			m_Renderer.renderFrame();
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}

		glfwSwapBuffers(m_MainWindow);
	}
}

void Application::initialize()
{
	fprintf(stdout, "initializing app\n");
	glfwSetErrorCallback(glfw_error_callback);

	if (!glfwInit()) exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	const char* glsl_version = "#version 460";

	m_width = m_height = 600;
	m_MainWindow = glfwCreateWindow(m_width, m_height, "MainWindow", NULL, NULL);

	if (!m_MainWindow) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(m_MainWindow);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glfwSwapInterval(1);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui::StyleColorsDark();

	// Setup Platform/Renderer backends
	ImGui_ImplOpenGL3_Init(glsl_version);
	ImGui_ImplGlfw_InitForOpenGL(m_MainWindow, true);
}

void Application::close()
{
	fprintf(stdout, "closing app\n");

	ImGui_ImplGlfw_Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_MainWindow);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}