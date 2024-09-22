#pragma once
#include <CelestiumPT.hpp>

#include "ModelImporter.hpp"
#include "Camera.hpp"
#include "Mesh.hpp"

#include <cstdint>
#include <vector>

struct PhsyicsSettings {
	bool enabled = true;
};

//TODO: rename; its like between a client project runtime and game procesing
class EditorSandbox {
public:
	EditorSandbox() = default;

	void initialise();
	void onUpdate(float delta_secs);
	void onRender(float delta_secs);
	void destroy();

	enum class RenderView {
		COMPOSITE = 0,
		NORMALS = 1,
		POSITIONS = 2,
		GAS = 3,
		UVs = 4,
		BARY = 5
	};

	constexpr static uint32_t minimum_viewport_height_threshold = 5;//any lower than this is prone to invalid render buffer sizes
	constexpr static uint32_t minimum_viewport_height = 10;
	constexpr static uint32_t viewport_vertical_scrolloverdraw_compensation_offset = 12;

	RenderView curent_renderview = RenderView::COMPOSITE;

	HostScene* m_HostSceneHandle = nullptr;
	Mesh m_selected_mesh;
	Camera m_Camera;

	PhsyicsSettings m_PhysCFG;

	std::vector<Mesh> m_Meshes;
	Renderer m_Renderer;
	GASBuilder m_GASBuilder;
	ModelImporter m_ModelImporter;
};