#include "HostScene.hpp"
#include "DeviceScene.cuh"
#include "DeviceMesh.cuh"

#include "Triangle.cuh"

//#include "Ray.cuh"
//#include "ShapeIntersection.cuh"
#include "SceneGeometry.cuh"
//#include "SceneGeometry.cuh"

#define CLSPT_NAME_STRING_LENGTH 32

static void setName(char* dst, const char* src) {
	memset(dst, 0, CLSPT_NAME_STRING_LENGTH);
	strncpy(dst, src, CLSPT_NAME_STRING_LENGTH);
	dst[CLSPT_NAME_STRING_LENGTH - 1] = '\0';
};

DeviceScene::DeviceScene(SceneGeometry* device_scene_geo) :DeviceSceneGeometry(device_scene_geo)
{
	if (DeviceSceneGeometry == nullptr) {
		cudaMallocManaged(&DeviceSceneGeometry, sizeof(DeviceSceneGeometry));
	}
	syncDeviceGeometry();
};

void DeviceScene::syncDeviceGeometry()
{
	if (DeviceSceneGeometry == nullptr)return;

	DeviceSceneGeometry->DeviceTrianglesBuffer = thrust::raw_pointer_cast(DeviceTriangles.data());
	DeviceSceneGeometry->DeviceTrianglesCount = DeviceTriangles.size();

	DeviceSceneGeometry->DeviceLightsBuffer = thrust::raw_pointer_cast(DeviceLights.data());
	DeviceSceneGeometry->DeviceLightsCount = DeviceLights.size();

	DeviceSceneGeometry->DeviceBVHTriangleIndicesBuffer = thrust::raw_pointer_cast(DeviceBVHTriangleIndices.data());
	DeviceSceneGeometry->DeviceBVHTriangleIndicesCount = DeviceBVHTriangleIndices.size();

	DeviceSceneGeometry->DeviceBVHNodesBuffer = thrust::raw_pointer_cast(DeviceBVHNodes.data());
	DeviceSceneGeometry->DeviceBVHNodesCount = DeviceBVHNodes.size();

	DeviceSceneGeometry->DeviceTLASNodesBuffer = thrust::raw_pointer_cast(DeviceTLASNodes.data());
	DeviceSceneGeometry->DeviceTLASNodesCount = DeviceTLASNodes.size();

	DeviceSceneGeometry->DeviceBLASesBuffer = thrust::raw_pointer_cast(DeviceBLASes.data());
	DeviceSceneGeometry->DeviceBLASesCount = DeviceBLASes.size();

	DeviceSceneGeometry->DeviceMeshesBuffer = thrust::raw_pointer_cast(DeviceMeshes.data());
	DeviceSceneGeometry->DeviceMeshesCount = DeviceMeshes.size();
};

HostScene::HostScene(DeviceScene* device_scene)
{
	m_DeviceScene = device_scene;
}

void HostScene::syncDeviceGeometry()
{
	m_DeviceScene->syncDeviceGeometry();
}

size_t HostScene::getTrianglesCount() {
	return m_DeviceScene->DeviceTriangles.size();
}

size_t HostScene::getMeshesCount() {
	return m_DeviceScene->DeviceMeshes.size();
}

size_t HostScene::getCamerasCount()
{
	return m_DeviceScene->DeviceCameras.size();
}

void HostScene::addCamera(HostCamera camera)
{
	DeviceCamera dcam;
	dcam.FOV_y_radians = camera.FOV_y_radians;
	dcam.invViewMatrix = Mat4(camera.m_transform);

	m_DeviceScene->DeviceCameras.push_back(dcam);
}

void HostScene::AddTriangle(
	glm::vec3 v0p, glm::vec3 v0n, glm::vec2 v0uv,
	glm::vec3 v1p, glm::vec3 v1n, glm::vec2 v1uv,
	glm::vec3 v2p, glm::vec3 v2n, glm::vec2 v2uv,
	glm::vec3 f_nrm, bool skip_sync)
{
	Triangle tri(
		Vertex(v0p, v0n, v0uv),
		Vertex(v1p, v1n, v1uv),
		Vertex(v2p, v2n, v2uv),
		f_nrm
	);
	m_DeviceScene->DeviceTriangles.push_back(tri);

	if (!skip_sync) {
		m_DeviceScene->syncDeviceGeometry();
	}
}

void HostScene::addLight(int triangle_idx, glm::vec3 color, float scale)
{
	Light dlight(&(m_DeviceScene->DeviceTriangles[triangle_idx]), make_float3(color.x, color.y, color.z), scale);
	(m_DeviceScene->DeviceTriangles[triangle_idx]).LightIdx = m_DeviceScene->DeviceLights.size();
	m_DeviceScene->DeviceLights.push_back(dlight);
}

void HostScene::AddMesh(HostMesh hmesh)
{
	DeviceMesh dmesh;
	setName(dmesh.name, hmesh.name.c_str());
	dmesh.triangle_offset_idx = hmesh.triangle_offset_idx;
	dmesh.tri_count = hmesh.tri_count;
	hmesh.modelMatrix = glm::inverse(hmesh.modelMatrix);
	Mat4 invmat(
		hmesh.modelMatrix[0][0], hmesh.modelMatrix[0][1], hmesh.modelMatrix[0][2], hmesh.modelMatrix[0][3],  // First column
		hmesh.modelMatrix[1][0], hmesh.modelMatrix[1][1], hmesh.modelMatrix[1][2], hmesh.modelMatrix[1][3],  // Second column
		hmesh.modelMatrix[2][0], hmesh.modelMatrix[2][1], hmesh.modelMatrix[2][2], hmesh.modelMatrix[2][3],  // Third column
		hmesh.modelMatrix[3][0], hmesh.modelMatrix[3][1], hmesh.modelMatrix[3][2], hmesh.modelMatrix[3][3]   // Fourth column
	);
	dmesh.inverseModelMatrix = invmat;
	m_DeviceScene->DeviceMeshes.push_back(dmesh);
	m_DeviceScene->syncDeviceGeometry();
}

void HostScene::LogStatus()
{
	DeviceScene* dscene = m_DeviceScene;

	printf("dev prim indices\n");
	for (int idx = 0; idx < dscene->DeviceBVHTriangleIndices.size(); idx++) {
		int n = dscene->DeviceBVHTriangleIndices[idx];
		printf("%d, ", n);
	}
	printf("\n");
	printf("bvh leafnodes prim indices\n");
	for (int idx = 0; idx < dscene->DeviceBVHNodes.size(); idx++) {
		if (dscene->DeviceBVHNodes[idx].triangle_indices_count < 1)continue;
		int o = dscene->DeviceBVHNodes[idx].left_child_or_triangle_indices_start_idx;
		int c = dscene->DeviceBVHNodes[idx].triangle_indices_count;
		printf("leafnode %d: offset=%d, count=%d,", idx, o, c);
	}
	printf("\n");
	printf("blas bvh indices\n");
	for (int idx = 0; idx < dscene->DeviceBLASes.size(); idx++) {
		int r = dscene->DeviceBLASes[idx].m_BVHRootIdx;
		int c = dscene->DeviceBLASes[idx].m_BVHNodesCount;
		int s = dscene->DeviceBLASes[idx].m_BVHNodesStartIdx;
		printf("blas %d: node_root=%d, node_count=%d, node_start=%d,\n", idx, r, c, s);
		float3 min = dscene->DeviceBLASes[idx].m_BoundingBox.pMin, max = dscene->DeviceBLASes[idx].m_BoundingBox.pMax;
		printf(" bbox min x:%.3f y:%.3f z:%.3f | max x:%.3f y:%.3f z:%.3f\n", min.x, min.y, min.z, max.x, max.y, max.z);
	}
	printf("\n");
	printf("tlasNodes bvh indices\n");
	for (int idx = 0; idx < dscene->DeviceTLASNodes.size(); idx++) {
		int bid = dscene->DeviceTLASNodes[idx].BLAS_idx;
		int l = (dscene->DeviceTLASNodes[idx].leftRight) & 0xFFFF;
		int r = (dscene->DeviceTLASNodes[idx].leftRight >> 16) & 0xFFFF;
		printf("tlas %d: blas_idx=%d, left_idx=%d, right_idx=%d,\n", idx, bid, l, r);
		float3 min = dscene->DeviceTLASNodes[idx].m_BoundingBox.pMin, max = dscene->DeviceTLASNodes[idx].m_BoundingBox.pMax;
		printf(" bbox min x:%.3f y:%.3f z:%.3f | max x:%.3f y:%.3f z:%.3f\n", min.x, min.y, min.z, max.x, max.y, max.z);
	}
	printf("\n");
	printf("TLAS RootIdx %d\n", dscene->DeviceSceneGeometry->GAS_structure.tlas.m_TLASRootIdx);
	//float3 max = dscene->DeviceSceneGeometry->GAS_structure.tlas.m_TLASRootIdx, min = dscene->DeviceSceneGeometry->GAS_structure.tlas.m_BoundingBox.pMin;
	//printf("TLAS bbox min x:%.3f y:%.3f z:%.3f | max x:%.3f y:%.3f z:%.3f\n", );
	printf("TLAS nodes count %d\n", dscene->DeviceSceneGeometry->GAS_structure.tlas.m_TLASnodesCount);
	printf("TLAS BLAS count %d\n", dscene->DeviceSceneGeometry->GAS_structure.tlas.m_BLASCount);
}

HostMesh HostScene::getMesh(size_t mesh_idx)
{
	assert(mesh_idx < m_DeviceScene->DeviceMeshes.size(), "DeviceMesh access Out Of Bounds");//TODO:fix

	DeviceMesh* dmeshptr = thrust::raw_pointer_cast(&m_DeviceScene->DeviceMeshes[mesh_idx]);
	HostMesh hmesh(dmeshptr);
	return hmesh;
}