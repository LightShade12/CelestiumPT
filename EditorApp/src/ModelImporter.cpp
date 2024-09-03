#include "ModelImporter.hpp"
#include "glm/glm.hpp"
#include <iostream>

static std::string GetFilePathExtension(const std::string& FileName) {
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

bool ModelImporter::loadGLTF(const char* filepath, HostScene* scene_object)
{
	m_WorkingScene = scene_object;
	bool status = false;
	status = loadGLTFModel(filepath);
	if (!status)return false;

	//load textures
	//load materials

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
	for (size_t nodeIdx = 0; nodeIdx < m_SceneModel.nodes.size(); nodeIdx++)
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
	printf("parse mesh called\n");
	tinygltf::Mesh gltf_mesh = m_SceneModel.meshes[mesh_node.mesh];

	std::vector<glm::vec3> loadedMeshPositions;
	std::vector<glm::vec3>loadedMeshNormals;
	std::vector<glm::vec2>loadedMeshUVs;
	std::vector<int>loadedMeshPrimitiveMatIdx;

	//Mesh drt_mesh;
	//setName(drt_mesh.Name, gltf_mesh.name.c_str());
	//printf("\nprocessing mesh:%s\n", gltf_mesh.name.c_str());

	//drt_mesh.m_primitives_offset = m_WorkingScene->getTrianglesCount();
	HostMesh mesh;
	mesh.triangle_offset_idx = m_WorkingScene->getTrianglesCount();
	extractVertices(gltf_mesh, loadedMeshPositions,
		loadedMeshNormals, loadedMeshUVs, loadedMeshPrimitiveMatIdx);
	mesh.tri_count = loadedMeshPositions.size() / 3;
	if (mesh_node.matrix.size() > 0) {
		mesh.setTransform(glm::mat4(
			mesh_node.matrix[0], mesh_node.matrix[1], mesh_node.matrix[2], mesh_node.matrix[3],
			mesh_node.matrix[4], mesh_node.matrix[5], mesh_node.matrix[6], mesh_node.matrix[7],
			mesh_node.matrix[8], mesh_node.matrix[9], mesh_node.matrix[10], mesh_node.matrix[11],
			mesh_node.matrix[12], mesh_node.matrix[13], mesh_node.matrix[14], mesh_node.matrix[15]
		));
	}
	else {
		mesh.setTransform(glm::mat4(1));
	}

	m_WorkingScene->AddMesh(mesh);

	printf("mesh positions: %zu\nmesh normals: %zu\n", loadedMeshPositions.size(), loadedMeshNormals.size());
	//Contruct and push Triangles
	//Positions.size() and vertex_normals.size() must be equal!
	//TODO: error handling ^

	for (size_t i = 0; i < loadedMeshPositions.size(); i += 3)
	{
		//surface normal construction
		glm::vec3 p0 = loadedMeshPositions[i + 1] - loadedMeshPositions[i];
		glm::vec3 p1 = loadedMeshPositions[i + 2] - loadedMeshPositions[i];
		glm::vec3 faceNormal = cross(p0, p1);

		glm::vec3 avgVertexNormal = (loadedMeshNormals[i] + loadedMeshNormals[i + 1] + loadedMeshNormals[i + 2]) / 3.f;
		float ndot = dot(faceNormal, avgVertexNormal);

		glm::vec3 surface_normal = (ndot < 0.0f) ? -faceNormal : faceNormal;

		//uint32_t mtidx = loadedMeshPrimitiveMatIdx[i / 3];

		m_WorkingScene->AddTriangle(
			loadedMeshPositions[i], loadedMeshNormals[i],
			loadedMeshPositions[i + 1], loadedMeshNormals[i + 1],
			loadedMeshPositions[i + 2], loadedMeshNormals[i + 2],
			normalize(surface_normal)
		);
		//DustRayTracer::HostMaterial mat = m_WorkingScene->getMaterial(mtidx);
		//glm::vec3 emcol = mat.getEmissiveColor();
		//if (mat.getEmissionTextureIndex() >= 0 ||
		//	!(emcol.x == 0 && emcol.y == 0 && emcol.z == 0)) {
		//	m_WorkingScene->addTriangleLightidx(m_WorkingScene->getTrianglesBufferSize() - 1);
		//}
	}

	//m_WorkingScene->addMesh(drt_mesh);
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
	return false;
}

bool ModelImporter::parseCamera(tinygltf::Node camera_node)
{
	/*tinygltf::Camera gltf_camera = m_SceneModel.cameras[gltf_node.camera];
		printf("\nfound a camera: %s\n", gltf_camera.name.c_str());
		glm::vec3 cpos = { gltf_node.translation[0] ,gltf_node.translation[1] ,gltf_node.translation[2] };
		DustRayTracer::HostCamera drt_camera;
		setName(drt_camera.getNamePtr(), gltf_camera.name.c_str());

		drt_camera.setPosition(glm::vec3(cpos.x, cpos.y, cpos.z));
		drt_camera.setVerticalFOV(gltf_camera.perspective.yfov);

		if (gltf_node.rotation.size() > 0) {
			float qx = gltf_node.rotation[0];
			float qy = gltf_node.rotation[1];
			float qz = gltf_node.rotation[2];
			float qw = gltf_node.rotation[3];
			glm::quat quaternion(qw, qx, qy, qz);
			glm::mat4 rotationMatrix = glm::toMat4(quaternion);
			glm::vec3 forwardDir = -glm::vec3(rotationMatrix[2]);
			glm::vec3 lookDir = glm::vec3(forwardDir.x, forwardDir.y, forwardDir.z);
			drt_camera.setLookDir(glm::vec3(lookDir.x, lookDir.y, lookDir.z));
		}

		m_WorkingScene->addCamera(drt_camera);*/
	return false;
}

bool ModelImporter::parseNode(tinygltf::Node node)
{
	return false;
}

bool ModelImporter::loadTextures(const tinygltf::Model& model, bool is_binary)
{
	return false;
}