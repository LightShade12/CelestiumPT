#include "EditorSandbox.hpp"
#include "Application.hpp"

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <glad/include/glad/glad.h>
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"
#include "imgui/imgui.h"

bool processMouse(GLFWwindow* window, Camera* camera, float delta_ts);

void EditorSandbox::initialise()
{
	m_HostSceneHandle = m_Renderer.getCurrentScene();//non owning; empty-initialized scene structure

	m_ModelImporter.loadGLTF("../models/moving_test.glb", m_HostSceneHandle);//uses host API to add scene geo
	//TODO: make loading automatically sync geometry
	m_HostSceneHandle->syncDeviceGeometry();//updates raw buffer data; not needed at this point

	m_GASBuilder.build(m_HostSceneHandle);
	m_HostSceneHandle->syncDeviceGeometry();

	//m_HostSceneHandle->LogStatus();

	if (m_HostSceneHandle->getMeshesCount() > 0)
		m_selected_mesh = Mesh(m_HostSceneHandle->getMesh(m_HostSceneHandle->getMeshesCount() - 1));

	for (size_t obj_idx = 0; obj_idx < m_HostSceneHandle->getMeshesCount(); obj_idx++) {
		m_Meshes.push_back(Mesh(m_HostSceneHandle->getMesh(obj_idx)));
	}

	m_Camera = Camera(m_Renderer.getCurrentCamera());
}

void EditorSandbox::destroy()
{
}

static bool s_updateCam = false;
static bool s_updateMesh = false;

void EditorSandbox::onUpdate(float delta)
{
	s_updateCam |= processMouse(Application::Get().getWindowHandle(), &m_Camera, delta);

	if (m_PhysCFG.enabled) {
		for (Mesh mesh : m_Meshes)
		{
			bool updatemesh = false;
			glm::vec3 translation(0);
			glm::vec3 rotation(0);
			if (mesh.host_mesh_handle.name == "monkey") {
				rotation.y = mesh.rotation.y + (10 * glfwGetTime());
				updatemesh |= true;
			}
			if (mesh.host_mesh_handle.name == "moving_platform") {
				translation.z = mesh.translation.z + (1 * sinf(glfwGetTime()));
				updatemesh |= true;
			}
			if (mesh.host_mesh_handle.name == "gear") {
				translation.x = mesh.translation.x + (1 * sinf(glfwGetTime()));
				updatemesh |= true;
			}
			if (mesh.host_mesh_handle.name == "teapot") {
				translation.y = mesh.translation.y + (0.15 * sinf(glfwGetTime()));
				updatemesh |= true;
			}

			if (updatemesh) {
				glm::mat4 trans = (glm::translate(glm::mat4(1), translation));
				glm::mat4 scale = (glm::scale(glm::mat4(1), mesh.scale));
				glm::mat4 rot_x = glm::rotate(glm::mat4(1), glm::radians(rotation.x), glm::vec3(1, 0, 0));
				glm::mat4 rot_y = glm::rotate(glm::mat4(1), glm::radians(rotation.y), glm::vec3(0, 1, 0));
				glm::mat4 rot_z = glm::rotate(glm::mat4(1), glm::radians(rotation.z), glm::vec3(0, 0, 1));
				glm::mat4 model = mesh.original_tranform * trans * rot_x * rot_y * rot_z * scale;
				mesh.host_mesh_handle.setTransform(model);
				mesh.host_mesh_handle.updateDevice(m_Renderer.getCurrentScene());
				m_Renderer.clearAccumulation();
			}
		}
	}

	if (s_updateMesh) {
		glm::mat4 trans = (glm::translate(glm::mat4(1),
			glm::vec3(m_selected_mesh.translation.x, m_selected_mesh.translation.y, m_selected_mesh.translation.z)));
		glm::mat4 scale = (glm::scale(glm::mat4(1), m_selected_mesh.scale));
		glm::mat4 rot_x = glm::rotate(glm::mat4(1), glm::radians(m_selected_mesh.rotation.x), glm::vec3(1, 0, 0));
		glm::mat4 rot_y = glm::rotate(glm::mat4(1), glm::radians(m_selected_mesh.rotation.y), glm::vec3(0, 1, 0));
		glm::mat4 rot_z = glm::rotate(glm::mat4(1), glm::radians(m_selected_mesh.rotation.z), glm::vec3(0, 0, 1));
		glm::mat4 model = m_selected_mesh.original_tranform * trans * rot_x * rot_y * rot_z * scale;
		//print_matrix(model);
		m_selected_mesh.host_mesh_handle.setTransform(model);
		m_selected_mesh.host_mesh_handle.updateDevice(m_Renderer.getCurrentScene());
		m_Renderer.clearAccumulation();
	}

	if (s_updateCam)
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

	s_updateCam = false;
	s_updateMesh = false;
}

void EditorSandbox::onRender(float delta_secs)
{
	{
		ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

		ImGui::Begin("Dev Window");
		ImGui::Text("FPS: %.2f", 1000.f / (delta_secs * 1000.f));//TODO: fix timing mterics
		ImGui::Text("Delta time: %.3fms", (delta_secs * 1000.f));
		ImGui::Text("Loaded meshes: %zu", m_Renderer.getCurrentScene()->getMeshesCount());
		ImGui::Text("Loaded triangles: %zu", m_Renderer.getCurrentScene()->getTrianglesCount());
		ImGui::Separator();
		if (ImGui::BeginTabBar("dev_tabs")) {
			if (ImGui::BeginTabItem("Rendering")) {
				if (ImGui::CollapsingHeader("Debug")) {
					ImGui::Combo("Renderer mode", (int*)&curent_renderview,
						"Composite\0Normals\0Positions\0GAS Debug");
				};

				if (ImGui::CollapsingHeader("Camera")) {
					ImGui::Text("Camera transformations");
					s_updateCam |= ImGui::DragFloat3("Camera translation", &m_Camera.position.x);
				};
				if (ImGui::CollapsingHeader("Pathtracing")) {
					ImGui::Checkbox("Accumulation", &(m_Renderer.getIntegratorSettings()->accumulate));
					ImGui::InputInt("Ray bounces", &(m_Renderer.getIntegratorSettings()->bounces));
				};
				if (ImGui::CollapsingHeader("Geometry")) {
					ImGui::Text("Mesh transformations");
					{
						static int sel_idx = 0;
						if (ImGui::InputInt("selected mesh idx", &sel_idx)) {
							sel_idx = (sel_idx < 0) ? 0 :
								(sel_idx == m_Renderer.getCurrentScene()->getMeshesCount()) ? (m_Renderer.getCurrentScene()->getMeshesCount() - 1) : sel_idx;
							m_selected_mesh = Mesh(m_Renderer.getCurrentScene()->getMesh(sel_idx));
						}
					}
					s_updateMesh |= ImGui::DragFloat3("Translation", &m_selected_mesh.translation.x, 0.02);
					s_updateMesh |= ImGui::DragFloat3("Scale", &m_selected_mesh.scale.x, 0.05);
					s_updateMesh |= ImGui::DragFloat3("Rotation(degrees)", &m_selected_mesh.rotation.x, 0.2);
				};
				if (ImGui::CollapsingHeader("Post-Processing")) {
				};
				if (ImGui::CollapsingHeader("Denoising")) {
				};
				if (ImGui::CollapsingHeader("General")) {
				};

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Setup")) {
				if (ImGui::CollapsingHeader("Material")) {
				};
				if (ImGui::CollapsingHeader("Sky")) {
				};
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Physics")) {
				if (ImGui::CollapsingHeader("General")) {
					ImGui::Checkbox("Physics enabled", &m_PhysCFG.enabled);
				};
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		};
		ImGui::End();

		//-----------------------------------------------------------------------------
		ImVec2 vpdims;
		{
			ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);

			vpdims = ImGui::GetContentRegionAvail();

			if (curent_renderview == RenderView::COMPOSITE) {
				if (m_Renderer.getCompositeRenderTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getCompositeRenderTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(), (float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::NORMALS) {
				if (m_Renderer.getNormalsTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getNormalsTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(), (float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::POSITIONS) {
				if (m_Renderer.getNormalsTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getPositionsTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(), (float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::GAS) {
				if (m_Renderer.getGASDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getGASDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(), (float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			ImGui::End();
		}

		if (vpdims.y > 14)vpdims.y -= 12;//TODO: make this sensible var; not a constant
		if (vpdims.y < 5)vpdims.y = 10;
		m_Renderer.resizeResolution((int)vpdims.x, (int)vpdims.y);
		m_GASBuilder.refresh(m_Renderer.getCurrentScene());
		m_Renderer.renderFrame();
	}
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