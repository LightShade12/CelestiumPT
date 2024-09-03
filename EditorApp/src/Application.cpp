#include "Application.hpp"
#include "ModelImporter.hpp"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include <glad/include/glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <glfw/include/GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <iostream>

static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void print_matrix(const glm::mat4& mat) {
	printf("| %.3f %.3f %.3f %.3f |\n", mat[0].x, mat[1].x, mat[2].x, mat[3].x);
	printf("| %.3f %.3f %.3f %.3f |\n", mat[0].y, mat[1].y, mat[2].y, mat[3].y);
	printf("| %.3f %.3f %.3f %.3f |\n", mat[0].z, mat[1].z, mat[2].z, mat[3].z);
	printf("| %.3f %.3f %.3f %.3f |\n\n\n", mat[0].w, mat[1].w, mat[2].w, mat[3].w);
}

bool processMouse(GLFWwindow* window, Camera* camera, float delta_ts);

Application::Application() : m_Camera()
{
	initialize();

	HostScene* hostscenehandle = m_Renderer.getCurrentScene();//non owning
	ModelImporter importer;
	importer.loadGLTF("../models/test_scene.glb", hostscenehandle);
	hostscenehandle->syncDeviceGeometry();

	m_Camera = Camera(m_Renderer.getCurrentCamera());
}

Application::~Application()
{
	close();
}

float deltaTime = 0.0f;	// Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

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
			ImGui::Text("Loaded meshes: %zu", m_Renderer.getCurrentScene()->getMeshesCount());
			ImGui::Text("Loaded triangles: %zu", m_Renderer.getCurrentScene()->getTrianglesCount());
			bool updateCam = false;

			updateCam |= ImGui::DragFloat3("Camera translation", &m_Camera.position.x);

			updateCam |= processMouse(m_MainWindow, &m_Camera, deltaTime);
			if (updateCam)
			{
				glm::mat4 view = glm::mat4(
					glm::vec4(m_Camera.right, 0),
					glm::vec4(m_Camera.up, 0),
					glm::vec4(m_Camera.forward, 0),
					glm::vec4(m_Camera.position, 1)
				);
				m_Camera.host_camera_handle->setTransform(view);
				m_Camera.host_camera_handle->updateDevice();
				m_Renderer.clearAccumulation();
			};
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

		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

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

glm::vec2 lastmousepos = { 0,0 };

bool processMouse(GLFWwindow* window, Camera* camera, float delta_ts)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	glm::vec2 mousePos = { xpos,ypos };
	glm::vec2 delta = (mousePos - lastmousepos) * 0.002f;
	lastmousepos = mousePos;

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		return false;
	}

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	bool moved = false;

	constexpr glm::vec3 global_up(0, 1, 0);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		camera->position += camera->movement_speed * delta_ts * camera->forward; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		camera->position -= camera->movement_speed * delta_ts * camera->forward; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		camera->position -= camera->movement_speed * delta_ts * camera->right; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		camera->position += camera->movement_speed * delta_ts * camera->right; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		camera->position += camera->movement_speed * delta_ts * global_up; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		camera->position -= camera->movement_speed * delta_ts * global_up; moved |= true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * camera->rot_speed;
		float yawDelta = delta.x * camera->rot_speed;

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, camera->right),
			glm::angleAxis(-yawDelta, global_up)));

		camera->forward = glm::normalize(glm::rotate(q, camera->forward));
		camera->right = normalize(glm::cross(camera->forward, global_up));
		camera->up = normalize(glm::cross(camera->right, camera->forward));

		moved = true;
	}

	return moved;
}