#pragma once
#include "celestium_pt.hpp"

#include "model_importer.hpp"
#include "camera.hpp"
#include "mesh.hpp"

#include <cstdint>
#include <vector>

struct PhsyicsSettings {
	bool enabled = false;
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
		BARY = 5,
		OBJECTID = 6,
		LOCALPOSITION = 7,
		VELOCITY = 8,
		DEPTH = 9,
		ALBEDO = 10,
		VARIANCE = 11,
		HEATMAP = 12,
		BBOXHEATMAP = 13,
		SPARSE_GRAD = 14
	};

private:
	void saveImagePNG();
public:
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