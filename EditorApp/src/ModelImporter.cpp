#include "ModelImporter.hpp"
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <iostream>

static std::string GetFilePathExtension(const std::string& FileName) {
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

//TODO:more coordinated loading

bool ModelImporter::loadGLTF(const char* filepath, HostScene* scene_object)
{
	m_WorkingScene = scene_object;
	bool status = false;
	status = loadGLTFModel(filepath);
	if (!status)return false;

	//load textures
	//load materials
	loadMaterials(m_SceneModel);

	for (std::string extensionname : m_SceneModel.extensionsUsed) {
		printf("using: %s\n", extensionname.c_str());
	}
	for (std::string extensionname : m_SceneModel.extensionsRequired) {
		printf("required: %s\n", extensionname.c_str());
	}
	printf("Detected nodes in file:%zu\n", m_SceneModel.nodes.size());
	printf("Detected meshes in file:%zu\n", m_SceneModel.meshes.size());
	printf("Detected cameras in file:%zu\n", m_SceneModel.cameras.size());

	//node looping
	for (size_t nodeIdx = 0; nodeIdx < m_SceneModel.scenes[0].nodes.size(); nodeIdx++)
	{
		tinygltf::Node gltf_node = m_SceneModel.nodes[m_SceneModel.scenes[0].nodes[nodeIdx]];
		printf("Processing node: %s\n", gltf_node.name.c_str());

		if (gltf_node.children.size() > 0) {
			parseNode(gltf_node);
		}

		if (gltf_node.camera >= 0) {
			parseCamera(gltf_node);
		}

		if (gltf_node.mesh >= 0) {
			parseMesh(gltf_node);
		}
	}

	//default fallback camera
	if (scene_object->getCamerasCount() < 1) {
		HostCamera hcam;
		hcam.FOV_y_radians = glm::radians(60.f);
		hcam.setTransform(
			glm::mat4(
				glm::vec4(1, 0, 0, 0),
				glm::vec4(0, 1, 0, 0),
				glm::vec4(0, 0, -1, 0),
				glm::vec4(0, 0, 0, 0)
			)
		);
		scene_object->addCamera(hcam);
	}

	return status;
}

bool ModelImporter::loadGLTFModel(const char* filename)
{
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;
	bool load_success = false;

	std::string ext = GetFilePathExtension(filename);

	//Load
	if (ext.compare("glb") == 0) {
		// assume binary glTF.
		load_success = loader.LoadBinaryFromFile(&m_SceneModel, &err, &warn, filename);
		is_binary_file = true;
	}
	else {
		// assume ascii glTF.
		load_success = loader.LoadASCIIFromFile(&m_SceneModel, &err, &warn, filename);
	}

	//Loader logging
	if (!warn.empty()) {
		std::cerr << "WARN: " << warn << std::endl;
	}
	if (!err.empty()) {
		std::cerr << "ERR: " << err << std::endl;
	}
	if (!load_success)
		std::cerr << "Failed to load glTF: " << filename << std::endl;
	else
		std::cerr << "Loaded glTF: " << filename << std::endl;

	return load_success;
}

bool ModelImporter::parseMesh(tinygltf::Node mesh_node)
{
	tinygltf::Mesh gltf_mesh = m_SceneModel.meshes[mesh_node.mesh];

	std::vector<glm::vec3> loadedMeshPositions;
	std::vector<glm::vec3>loadedMeshNormals;
	std::vector<glm::vec2>loadedMeshUVs;
	std::vector<int>loadedMeshPrimitiveMatIdx;

	printf("\nprocessing mesh:%s\n", mesh_node.name.c_str());

	HostMesh mesh;
	mesh.name = mesh_node.name;
	mesh.triangle_offset_idx = m_WorkingScene->getTrianglesCount();
	extractVertices(gltf_mesh, loadedMeshPositions,
		loadedMeshNormals, loadedMeshUVs, loadedMeshPrimitiveMatIdx);
	mesh.tri_count = loadedMeshPositions.size() / 3;

	glm::mat4 modelMatrix(1);
	if (mesh_node.matrix.size() > 0) {
		modelMatrix = glm::mat4(
			mesh_node.matrix[0], mesh_node.matrix[1], mesh_node.matrix[2], mesh_node.matrix[3],
			mesh_node.matrix[4], mesh_node.matrix[5], mesh_node.matrix[6], mesh_node.matrix[7],
			mesh_node.matrix[8], mesh_node.matrix[9], mesh_node.matrix[10], mesh_node.matrix[11],
			mesh_node.matrix[12], mesh_node.matrix[13], mesh_node.matrix[14], mesh_node.matrix[15]
		);
	}
	else {
		if (mesh_node.scale.size() > 0) {
			glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(mesh_node.scale[0], mesh_node.scale[1], mesh_node.scale[2]));
			modelMatrix = scaleMat * modelMatrix;
		}

		if (mesh_node.rotation.size() > 0) {
			glm::quat quaternion = glm::quat(mesh_node.rotation[3], mesh_node.rotation[0], mesh_node.rotation[1], mesh_node.rotation[2]);
			glm::mat4 rotMat = glm::toMat4(quaternion);
			modelMatrix = rotMat * modelMatrix;
		}

		if (mesh_node.translation.size() > 0) {
			glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), glm::vec3(mesh_node.translation[0], mesh_node.translation[1], mesh_node.translation[2]));
			modelMatrix = translationMat * modelMatrix;
		}
	}
	mesh.setTransform(modelMatrix);

	m_WorkingScene->addMesh(mesh);

	//TODO: error handling
	//Positions.size() and vertex_normals.size() must be equal!
	if (loadedMeshPositions.size() != loadedMeshNormals.size())printf("\n>> [POSITIONS-NORMALS COUNT MISMATCH] !\n");

	//Contruct and push Triangles
	for (size_t i = 0; i < loadedMeshPositions.size(); i += 3)
	{
		//geometric normal construction
		glm::vec3 edge0 = loadedMeshPositions[i + 1] - loadedMeshPositions[i];
		glm::vec3 edge1 = loadedMeshPositions[i + 2] - loadedMeshPositions[i];
		glm::vec3 geo_norm = cross(edge0, edge1);

		glm::vec3 avgVertexNormal = (loadedMeshNormals[i] + loadedMeshNormals[i + 1] + loadedMeshNormals[i + 2]) / 3.f;

		float shn_gn_dot = dot(geo_norm, avgVertexNormal);
		glm::vec3 geometric_normal = (shn_gn_dot < 0.0f) ? -geo_norm : geo_norm;

		int mtidx = loadedMeshPrimitiveMatIdx[i / 3];

		m_WorkingScene->AddTriangle(
			loadedMeshPositions[i], loadedMeshNormals[i], loadedMeshUVs[i],
			loadedMeshPositions[i + 1], loadedMeshNormals[i + 1], loadedMeshUVs[i + 1],
			loadedMeshPositions[i + 2], loadedMeshNormals[i + 2], loadedMeshUVs[i + 2],
			normalize(geometric_normal), mtidx
		);

		HostMaterial mat = m_WorkingScene->getMaterial(mtidx);

		if (!(mat.emission_color_factor.x == 0 && mat.emission_color_factor.y == 0 && mat.emission_color_factor.z == 0)) {
			//m_WorkingScene->addLight(m_WorkingScene->getTrianglesCount() - 1,
			//	mat.emission_color_factor, mat.emission_strength);
		}
	}

	//printf("\rloaded mesh:%zu/%zu", nodeIdx + 1, m_SceneModel.nodes.size());
	return false;
}

bool ModelImporter::extractVertices(tinygltf::Mesh mesh,
	std::vector<glm::vec3>& positions,
	std::vector<glm::vec3>& normals,
	std::vector<glm::vec2>& tex_coords,
	std::vector<int>& primitive_mat_idx)
{
	//printf("total primitives: %zu\n", mesh.primitives.size());
	for (size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++)
	{
		//printf("prim idx:%zu \n", primIdx);
		tinygltf::Primitive primitive = mesh.primitives[primIdx];

		int pos_attrib_accesorIdx = primitive.attributes["POSITION"];
		int nrm_attrib_accesorIdx = primitive.attributes["NORMAL"];
		int uv_attrib_accesorIdx = primitive.attributes["TEXCOORD_0"];

		int indices_accesorIdx = primitive.indices;

		tinygltf::Accessor pos_accesor = m_SceneModel.accessors[pos_attrib_accesorIdx];
		tinygltf::Accessor nrm_accesor = m_SceneModel.accessors[nrm_attrib_accesorIdx];
		tinygltf::Accessor uv_accesor = m_SceneModel.accessors[uv_attrib_accesorIdx];
		tinygltf::Accessor indices_accesor = m_SceneModel.accessors[indices_accesorIdx];

		int pos_accesor_byte_offset = pos_accesor.byteOffset;//redundant
		int nrm_accesor_byte_offset = nrm_accesor.byteOffset;//redundant
		int uv_accesor_byte_offset = uv_accesor.byteOffset;//redundant
		int indices_accesor_byte_offset = indices_accesor.byteOffset;//redundant

		tinygltf::BufferView pos_bufferview = m_SceneModel.bufferViews[pos_accesor.bufferView];
		tinygltf::BufferView nrm_bufferview = m_SceneModel.bufferViews[nrm_accesor.bufferView];
		tinygltf::BufferView uv_bufferview = m_SceneModel.bufferViews[uv_accesor.bufferView];
		tinygltf::BufferView indices_bufferview = m_SceneModel.bufferViews[indices_accesor.bufferView];

		int pos_buffer_byte_offset = pos_bufferview.byteOffset;
		int nrm_buffer_byte_offset = nrm_bufferview.byteOffset;
		int uv_buffer_byte_offset = uv_bufferview.byteOffset;

		tinygltf::Buffer indices_buffer = m_SceneModel.buffers[indices_bufferview.buffer];//should alawys be zero?

		//printf("normals accesor count: %d\n", nrm_accesor.count);
		//printf("positions accesor count: %d\n", pos_accesor.count);
		//printf("UVs accesor count: %d\n", uv_accesor.count);
		//printf("indices accesor count: %d\n", indices_accesor.count);

		unsigned short* indicesbuffer = (unsigned short*)(indices_buffer.data.data());
		glm::vec3* positions_buffer = (glm::vec3*)(indices_buffer.data.data() + pos_buffer_byte_offset);
		glm::vec3* normals_buffer = (glm::vec3*)(indices_buffer.data.data() + nrm_buffer_byte_offset);
		glm::vec2* UVs_buffer = (glm::vec2*)(indices_buffer.data.data() + uv_buffer_byte_offset);

		for (int i = (indices_bufferview.byteOffset / 2); i < (indices_bufferview.byteLength + indices_bufferview.byteOffset) / 2; i++)
		{
			positions.push_back(positions_buffer[indicesbuffer[i]]);
			normals.push_back(normals_buffer[indicesbuffer[i]]);
			tex_coords.push_back(UVs_buffer[indicesbuffer[i]]);
		}
		for (size_t i = 0; i < indices_accesor.count / 3; i++)//no of triangles per primitive
		{
			primitive_mat_idx.push_back(primitive.material);
		}
	}
	return true;
}

bool ModelImporter::loadMaterials(const tinygltf::Model& model)
{
	printf("detected materials count in file: %zu\n", model.materials.size());

	for (size_t matIdx = 0; matIdx < model.materials.size(); matIdx++)
	{
		tinygltf::Material gltf_material = model.materials[matIdx];
		printf("loading material: %s\n", gltf_material.name.c_str());
		tinygltf::PbrMetallicRoughness PBR_data = gltf_material.pbrMetallicRoughness;
		//setName(drt_material.getNamePtr(), gltf_material.name.c_str());
		glm::vec3 albedo_factor = glm::vec3(PBR_data.baseColorFactor[0], PBR_data.baseColorFactor[1], PBR_data.baseColorFactor[2]);//TODO: We just use RGB material albedo for now
		glm::vec3 emission_factor = glm::vec3(gltf_material.emissiveFactor[0], gltf_material.emissiveFactor[1], gltf_material.emissiveFactor[2]);
		float emission_strength = 0.f;
		//if (PBR_data.baseColorTexture.index >= 0)drt_material.setAlbedoTextureIndex(model.textures[PBR_data.baseColorTexture.index].source);
		//if (PBR_data.metallicRoughnessTexture.index >= 0)drt_material.setRoughnessTextureIndex(model.textures[PBR_data.metallicRoughnessTexture.index].source);
		//if (gltf_material.normalTexture.index >= 0)drt_material.setNormalTextureIndex(model.textures[gltf_material.normalTexture.index].source);
		//drt_material.setNormalMapScale(gltf_material.normalTexture.scale);
		//if (gltf_material.emissiveTexture.index >= 0)drt_material.setEmissionTextureIndex(model.textures[gltf_material.emissiveTexture.index].source);
		//drt_material.setMetallicity((PBR_data.metallicRoughnessTexture.index >= 0) ? 1.f : PBR_data.metallicFactor);
		//drt_material.setRoughness((PBR_data.metallicRoughnessTexture.index >= 0) ? 1.f : PBR_data.roughnessFactor);

		//if (gltf_material.extensions.find("KHR_materials_transmission") != gltf_material.extensions.end()) {
		//	drt_material.setTransmission(gltf_material.extensions["KHR_materials_transmission"].Get("transmissionFactor").GetNumberAsDouble());
		//};
		//if (gltf_material.extensions.find("KHR_materials_ior") != gltf_material.extensions.end()) {
		//	drt_material.setIOR(gltf_material.extensions["KHR_materials_ior"].Get("ior").GetNumberAsDouble());
		//};
		if (gltf_material.extensions.find("KHR_materials_emissive_strength") != gltf_material.extensions.end()) {
			emission_strength = (gltf_material.extensions["KHR_materials_emissive_strength"].Get("emissiveStrength").GetNumberAsDouble());
		};

		//printToConsole("albedo texture idx: %d\n", drt_material.AlbedoTextureIndex);
		m_WorkingScene->addMaterial(albedo_factor, emission_factor, emission_strength);
	}
	printf("loaded materials count: %zu \n\n", m_WorkingScene->getMaterialsCount());

	return true;
}

bool ModelImporter::parseCamera(tinygltf::Node camera_node)
{
	tinygltf::Camera gltf_camera = m_SceneModel.cameras[camera_node.camera];
	printf("\nfound a camera: %s\n", gltf_camera.name.c_str());

	HostCamera hcam;

	hcam.FOV_y_radians = gltf_camera.perspective.yfov;

	glm::mat4 viewMatrix(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, 1
	);

	if (camera_node.rotation.size() > 0) {
		glm::quat quaternion = glm::quat(camera_node.rotation[3], camera_node.rotation[0], camera_node.rotation[1], camera_node.rotation[2]);
		glm::mat4 rotmat = glm::toMat4(quaternion);
		viewMatrix = rotmat * viewMatrix;  // Apply rotation first
	}

	if (camera_node.translation.size() > 0) {
		glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), glm::vec3(camera_node.translation[0], camera_node.translation[1], camera_node.translation[2]));
		viewMatrix = translationMat * viewMatrix;
	}

	hcam.setTransform(viewMatrix);

	m_WorkingScene->addCamera(hcam);
	return false;
}

bool ModelImporter::parseNode(tinygltf::Node node)
{
	//node looping
	for (size_t nodeIdx = 0; nodeIdx < node.children.size(); nodeIdx++)
	{
		tinygltf::Node gltf_node = m_SceneModel.nodes[nodeIdx];
		printf("Processing node: %s\n", gltf_node.name.c_str());

		if (gltf_node.children.size() > 0) {
			parseNode(gltf_node);
		}

		if (gltf_node.camera >= 0) {
			parseCamera(gltf_node);
		}

		if (gltf_node.mesh >= 0) {
			parseMesh(gltf_node);
		}
	}
	return false;
}

bool ModelImporter::loadTextures(const tinygltf::Model& model, bool is_binary)
{
	return false;
}