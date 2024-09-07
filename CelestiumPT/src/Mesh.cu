#include "HostMesh.hpp"
#include "DeviceMesh.cuh"
#include "maths/maths_linear_algebra.cuh"

HostMesh::HostMesh(DeviceMesh* device_mesh)
{
	m_deviceMesh = device_mesh;
	Mat4 mat = m_deviceMesh->modelMatrix;
	modelMatrix = glm::mat4(
		mat[0].x, mat[0].y, mat[0].z, mat[0].w,
		mat[1].x, mat[1].y, mat[1].z, mat[1].w,
		mat[2].x, mat[2].y, mat[2].z, mat[2].w,
		mat[3].x, mat[3].y, mat[3].z, mat[3].w
	);
	triangle_offset_idx = device_mesh->triangle_offset_idx;
	tri_count = device_mesh->tri_count;
}

void HostMesh::updateDevice()
{
	if (m_deviceMesh != nullptr) {
		Mat4 mat(
			modelMatrix[0][0], modelMatrix[0][1], modelMatrix[0][2], modelMatrix[0][3],  // First column
			modelMatrix[1][0], modelMatrix[1][1], modelMatrix[1][2], modelMatrix[1][3],  // Second column
			modelMatrix[2][0], modelMatrix[2][1], modelMatrix[2][2], modelMatrix[2][3],  // Third column
			modelMatrix[3][0], modelMatrix[3][1], modelMatrix[3][2], modelMatrix[3][3]   // Fourth column
		);
		m_deviceMesh->modelMatrix = mat;
	}
}