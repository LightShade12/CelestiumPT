#pragma once
#include "CelestiumPT.hpp"
#include "tinygltf/tiny_gltf.h"

class ModelImporter
{
public:
	ModelImporter() = default;

	bool loadGLTF(const char* filepath, HostScene* scene_object);

private:
	HostScene* m_WorkingScene = nullptr;
	tinygltf::Model m_SceneModel;
	bool is_binary_file = false;

	bool extractVertices(tinygltf::Mesh mesh,
		std::vector<glm::vec3>& positions,
		std::vector<glm::vec3>& normals,
		std::vector<glm::vec2>& tex_coords,
		std::vector<int>& primitive_mat_idx);
	bool loadGLTFModel(const char* filename);
	bool loadMaterials(const tinygltf::Model& model);
	bool parseMesh(tinygltf::Node mesh_node);
	bool parseCamera(tinygltf::Node camera_node);
	bool parseNode(tinygltf::Node node);
	bool loadTextures(const tinygltf::Model& model, bool is_binary);
};