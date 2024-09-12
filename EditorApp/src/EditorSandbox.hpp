#pragma once
#include <CelestiumPT.hpp>

#include "ModelImporter.hpp"
#include "Camera.hpp"
#include "Mesh.hpp"

#include <vector>

struct PhsyicsSettings {
	bool enabled = true;
};

//TODO: rename; its like between a client project runtime and game procesing
class EditorSandbox {
public:
	EditorSandbox() = default;
	~EditorSandbox() = default;

	void initialise();
	void onUpdate(float delta);
	void onRender(float delta);
	void destroy();

	enum class RenderView {
		COMPOSITE = 0,
		NORMALS = 1,
		POSITIONS = 2,
		GAS = 3
	};
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