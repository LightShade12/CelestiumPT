#include "editor_sandbox.hpp"
#include "application.hpp"

#include "stb/stb_image_write.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/quaternion.hpp"

#include "glad/include/glad/glad.h"
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"
#include "imgui/imgui.h"

bool processMouse(GLFWwindow* window, Camera* camera, float delta_ts);
//TODO: investigate deprecated cuda library include warning: device_functions.cuh?
void EditorSandbox::initialise()
{
	m_HostSceneHandle = m_Renderer.getCurrentScene();//non owning; empty-initialized scene structure

	m_ModelImporter.loadGLTFfromFile("../models/cs16_dust_unit.glb", m_HostSceneHandle);//uses host API to add scene geo

	m_GASBuilder.build(m_HostSceneHandle);

	m_Renderer.setCamera(0);

	if (m_HostSceneHandle->getMeshesCount() > 0)
		m_selected_mesh = Mesh(m_HostSceneHandle->getMesh(0));

	for (size_t obj_idx = 0; obj_idx < m_HostSceneHandle->getMeshesCount(); obj_idx++) {
		m_Meshes.push_back(Mesh(m_HostSceneHandle->getMesh(obj_idx)));
	}

	m_Camera = Camera(m_Renderer.getCurrentCamera());
}

void EditorSandbox::destroy()
{
	printf("\ndestroy(): Sandbox closed\n");
}

void EditorSandbox::saveImagePNG()
{
	size_t w = m_Renderer.getFrameWidth(), h = m_Renderer.getFrameHeight();
	std::vector<GLubyte> frame_data(w * h * 4); // RGBA8

	// Save composite image
	glBindTexture(GL_TEXTURE_2D, m_Renderer.getCompositeRenderTargetTextureName());
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
	glBindTexture(GL_TEXTURE_2D, 0);
	stbi_flip_vertically_on_write(true);

	std::string composite_fname = "composite_img_" + std::to_string(m_Renderer.getSPP()) + "_spp.png";

	if (stbi_write_png(composite_fname.c_str(),
		w, h, 4, frame_data.data(), 4 * sizeof(GLubyte) * w) != 0)
		printf("\nComposite image saved: %s\n", composite_fname.c_str());
	else
		printf("\nComposite image save failed\n");

	// Save variance image
	glBindTexture(GL_TEXTURE_2D, m_Renderer.getIntegratedVarianceTargetTextureName());
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame_data.data());
	glBindTexture(GL_TEXTURE_2D, 0);

	std::string variance_fname = "variance_img_" + std::to_string(m_Renderer.getSPP()) + "_spp.png";

	if (stbi_write_png(variance_fname.c_str(),
		w, h, 4, frame_data.data(), 4 * sizeof(GLubyte) * w) != 0)
		printf("\nVariance image saved: %s\n", variance_fname.c_str());
	else
		printf("\nVariance image save failed\n");
}

static bool s_updateCam = false;
static bool s_updateMesh = false;

void EditorSandbox::onUpdate(float delta)
{
	s_updateCam |= processMouse(Application::Get().getWindowHandle(), &m_Camera, delta);

	if (!m_PhysCFG.enabled) {
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
			//if (mesh.host_mesh_handle.name == "Light") {
			//	translation.x = mesh.translation.x + (1.4 * sinf(glfwGetTime() * 1));
			//	translation.z = mesh.translation.x + (1.4 * cosf(glfwGetTime() * 1));
			//	updatemesh |= true;
			//}

			if (updatemesh) {
				glm::mat4 trans = (glm::translate(glm::mat4(1), translation));
				glm::mat4 scale = (glm::scale(glm::mat4(1), mesh.scale));
				glm::mat4 rot_x = glm::rotate(glm::mat4(1), glm::radians(rotation.x), glm::vec3(1, 0, 0));
				glm::mat4 rot_y = glm::rotate(glm::mat4(1), glm::radians(rotation.y), glm::vec3(0, 1, 0));
				glm::mat4 rot_z = glm::rotate(glm::mat4(1), glm::radians(rotation.z), glm::vec3(0, 0, 1));
				glm::mat4 model = mesh.original_tranform * trans * rot_x * rot_y * rot_z * scale;
				mesh.host_mesh_handle.setTransform(model);
				mesh.host_mesh_handle.updateDevice(m_Renderer.getCurrentScene());
				if (m_Renderer.getIntegratorSettings()->accumulate && !m_Renderer.getIntegratorSettings()->temporal_filter_enabled)
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
		if (m_Renderer.getIntegratorSettings()->accumulate && !m_Renderer.getIntegratorSettings()->temporal_filter_enabled)
			m_Renderer.clearAccumulation();
	}

	if (s_updateCam)
	{
		glm::mat4 inv_view = glm::mat4(
			glm::vec4(m_Camera.right, 0),
			glm::vec4(m_Camera.up, 0),
			glm::vec4(m_Camera.forward, 0),
			glm::vec4(m_Camera.position, 1)
		);

		m_Camera.host_camera_handle->setTransform(inv_view);
		m_Camera.host_camera_handle->updateDevice();
		if (m_Renderer.getIntegratorSettings()->accumulate && !m_Renderer.getIntegratorSettings()->temporal_filter_enabled)
			m_Renderer.clearAccumulation();
	};

	s_updateCam = false;
	s_updateMesh = false;
}

static int g_max_spp = 100;

void EditorSandbox::onRender(float delta_secs)
{
	{
		ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

		ImGui::Begin("Dev Window");
		ImGui::Text("FPS: %.2f", 1000.f / (delta_secs * 1000.f));
		ImGui::Text("Delta time: %.3fms", (delta_secs * 1000.f));
		ImGui::Text("Loaded meshes: %zu", m_Renderer.getCurrentScene()->getMeshesCount());
		ImGui::Text("Loaded triangles: %zu", m_Renderer.getCurrentScene()->getTrianglesCount());
		ImGui::Separator();
		if (ImGui::BeginTabBar("dev_tabs")) {
			if (ImGui::BeginTabItem("Rendering")) {
				if (ImGui::CollapsingHeader("Debug")) {
					ImGui::Text("SPP:%d\n", m_Renderer.getSPP());
					ImGui::SliderInt("max SPP", &g_max_spp, 1, 100);
					ImGui::Combo("Renderer view", (int*)&curent_renderview,
						"Composite\0Normals\0Positions\0GAS Debug\0UVs\0Barycentrics\0ObjectID\0LocalPosition\0Velocity\0Depth\0Albedo\0Variance\0Heatmap\0bbox heatmap\0sparse grads\0dense grads\0Misc debugview\0Mip0\0");
					if (curent_renderview == RenderView::GAS) {
						ImGui::SliderFloat("GAS shading brightness",
							&(m_Renderer.getIntegratorSettings()->GAS_shading_brightness), 0.0001, 0.1);
					}
				};

				if (ImGui::CollapsingHeader("Camera")) {
					ImGui::SeparatorText("Camera transformations");
					s_updateCam |= ImGui::DragFloat3("Camera translation", &m_Camera.position.x);
					if (ImGui::SliderAngle("FoV", &(m_Camera.fov_y_rad), 20, 120)) {
						m_Camera.recalculateProjection();
						s_updateCam |= true;
					};
					if (ImGui::SliderFloat("Exposure", &m_Camera.host_camera_handle->exposure, 0.f, 20, "%.3f",
						ImGuiSliderFlags_Logarithmic))
					{
						s_updateCam |= true;
					};
					ImGui::Checkbox("Auto Exposure", &(m_Renderer.getIntegratorSettings()->auto_exposure_enabled));
					if (m_Renderer.getIntegratorSettings()->auto_exposure_enabled) {
						ImGui::Indent();
						ImGui::SliderFloat("EV comp max", &(m_Renderer.getIntegratorSettings()->auto_exposure_max_comp), -20, 20);
						ImGui::SliderFloat("EV comp min", &(m_Renderer.getIntegratorSettings()->auto_exposure_min_comp), -20, 20);
						ImGui::SliderFloat("EV comp speed", &(m_Renderer.getIntegratorSettings()->auto_exposure_speed), 0, 1);
						ImGui::Unindent();
					}
					ImGui::SeparatorText("Bloom");
					ImGui::SliderFloat("Bloom lerp", &(m_Renderer.getIntegratorSettings()->bloom_lerp), 0, 1);
					ImGui::SliderFloat("Bloom internal lerp", &(m_Renderer.getIntegratorSettings()->bloom_internal_lerp), 0, 1);

					ImGui::SeparatorText("Motion");
					ImGui::SliderFloat("Speed", &m_Camera.movement_speed, 0, 10);
				};
				if (ImGui::CollapsingHeader("Pathtracing"))
				{
					ImGui::InputInt("Ray bounces", &(m_Renderer.getIntegratorSettings()->max_bounces));
					ImGui::Checkbox("Accumulation", &(m_Renderer.getIntegratorSettings()->accumulate));
					ImGui::SeparatorText("Denoising");
					ImGui::Checkbox("Temporal accumulation", &(m_Renderer.getIntegratorSettings()->temporal_filter_enabled));
					ImGui::Checkbox("SVGF denoiser", &(m_Renderer.getIntegratorSettings()->svgf_enabled));
					if (m_Renderer.getIntegratorSettings()->svgf_enabled) {
						ImGui::Indent();
						ImGui::Checkbox("Use 5x5 filter", &(m_Renderer.getIntegratorSettings()->use_5x5_filter));
						ImGui::Unindent();
					}
					ImGui::Checkbox("Adaptive filter", &(m_Renderer.getIntegratorSettings()->adaptive_temporal_filter_enabled));
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
				if (ImGui::CollapsingHeader("General")) {
					if (ImGui::Button("Save Frame as PNG")) {
						saveImagePNG();
					}
				};

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Setup")) {
				if (ImGui::CollapsingHeader("Material")) {
				};
				if (ImGui::CollapsingHeader("Sky"))
				{
					ImGui::Checkbox("SkyLight", &(m_Renderer.getIntegratorSettings()->skylight_enabled));
					ImGui::SliderFloat("SkyLight intensity", &(m_Renderer.getIntegratorSettings()->skylight_intensity), 0, 30);
					ImGui::Checkbox("SunLight", &(m_Renderer.getIntegratorSettings()->sunlight_enabled));
					ImGui::SliderFloat("SunLight intensity", &(m_Renderer.getIntegratorSettings()->sunlight_intensity), 0, 30);
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
			//TODO: set sensible storage for magic numbers
			ImGui::SetNextWindowSize(ImVec2(640 + 16, 360 + 47));
			ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar);

			vpdims = ImGui::GetContentRegionAvail();

			if (curent_renderview == RenderView::COMPOSITE) {
				if (m_Renderer.getCompositeRenderTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getCompositeRenderTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::NORMALS) {
				if (m_Renderer.getNormalsTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getNormalsTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::POSITIONS) {
				if (m_Renderer.getNormalsTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getPositionsTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::GAS) {
				if (m_Renderer.getGASDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getGASDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::UVs) {
				if (m_Renderer.getUVsDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getUVsDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::BARY) {
				if (m_Renderer.getBarycentricsDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getBarycentricsDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::OBJECTID) {
				if (m_Renderer.getObjectIDDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getObjectIDDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::LOCALPOSITION) {
				if (m_Renderer.getLocalPositionsTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getLocalPositionsTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::VELOCITY) {
				if (m_Renderer.getVelocityTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getVelocityTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::DEPTH) {
				if (m_Renderer.getDepthTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getDepthTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::ALBEDO) {
				if (m_Renderer.getAlbedoTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getAlbedoTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::VARIANCE) {
				if (m_Renderer.getIntegratedVarianceTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getIntegratedVarianceTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::ALBEDO) {
				if (m_Renderer.getAlbedoRenderTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getAlbedoRenderTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::HEATMAP) {
				if (m_Renderer.getHeatmapDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getHeatmapDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::BBOXHEATMAP) {
				if (m_Renderer.getBboxHeatmapDebugTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getBboxHeatmapDebugTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::SPARSE_GRAD) {
				if (m_Renderer.getSparseGradientTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getSparseGradientTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::DENSE_GRAD) {
				if (m_Renderer.getDenseGradientTargetTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getDenseGradientTargetTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::MISC_DBG) {
				if (m_Renderer.getMiscDebugTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getMiscDebugTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}
			else if (curent_renderview == RenderView::MIP0) {
				if (m_Renderer.getMip0DebugTextureName() != NULL)
					ImGui::Image((void*)(uintptr_t)m_Renderer.getMip0DebugTextureName(),
						ImVec2((float)m_Renderer.getFrameWidth(),
							(float)m_Renderer.getFrameHeight()), { 0,1 }, { 1,0 });
			}

			ImGui::BeginChild("viewport_status", ImVec2(ImGui::GetContentRegionAvail().x, 14), 0);

			//ImGui::SetCursorScreenPos({ ImGui::GetCursorScreenPos().x + 5, ImGui::GetCursorScreenPos().y + 4 });

			ImGui::Text("dims: %d x %d px", m_Renderer.getFrameWidth(), m_Renderer.getFrameHeight());
			ImGui::SameLine();
			ImGui::Text(" | RGBA32F");
			ImGui::EndChild();

			ImGui::End();
		}

		if (vpdims.y > viewport_vertical_scrolloverdraw_compensation_offset +
			minimum_viewport_height_threshold)vpdims.y -= viewport_vertical_scrolloverdraw_compensation_offset;
		if (vpdims.y < minimum_viewport_height_threshold)vpdims.y = minimum_viewport_height;

		m_Camera.resizeFrame((int)vpdims.x, (int)vpdims.y);
		m_Renderer.resizeResolution((int)vpdims.x, (int)vpdims.y);

		m_GASBuilder.refresh(m_Renderer.getCurrentScene());
		//if (m_Renderer.getSPP() < g_max_spp)
		m_Renderer.renderFrame();
		m_Camera.host_camera_handle->updateCamera();//Must happen per frame
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
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)//LEFT
	{
		camera->position -= camera->movement_speed * delta_ts * camera->right; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)//RIGHT
	{
		camera->position += camera->movement_speed * delta_ts * camera->right; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)//UP
	{
		camera->position += camera->movement_speed * delta_ts * global_up; moved |= true;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)//DOWN
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