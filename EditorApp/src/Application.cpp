#include "application.hpp"

#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE //glad loader instead of local gl
#include "glfw/include/GLFW/glfw3.h"

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static Application* s_Instance = nullptr;

Application::Application() : m_EditorSandbox()
{
	s_Instance = this;
	initialize();
}

Application& Application::Get()
{
	return *s_Instance;
}

Application::~Application()
{
	close();
	s_Instance = nullptr;
	printf("\ndestructed application");
}

static float delta_time_secs = 0.0f;
static float last_frame_secs = 0.0f;

void Application::run()
{
	fprintf(stdout, "start run..\n");
	//int width, height;
	glClearColor(0.f, 0.24f, 0.3f, 1.f);

	while (!glfwWindowShouldClose(m_MainWindow))
	{
		glfwPollEvents();
		m_EditorSandbox.onUpdate(delta_time_secs);

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

		m_EditorSandbox.onRender(delta_time_secs);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}

		//printf("secs: %.3f\n", (float)glfwGetTime());

		float currentFrame_secs = glfwGetTime();
		delta_time_secs = currentFrame_secs - last_frame_secs;

		//printf("delta secs: %.3f\n", (float)delta_time_secs);

		last_frame_secs = currentFrame_secs;

		glfwSwapBuffers(m_MainWindow);
	}
}

float Application::getDeltaTimeSeconds()
{
	return delta_time_secs;
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

	m_width = 640 + 16;
	m_height = 700;
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

	m_EditorSandbox.initialise();
}

void Application::close()
{
	fprintf(stdout, "closing app\n");

	m_EditorSandbox.destroy();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_MainWindow);
	glfwTerminate();
}