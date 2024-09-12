#include "HostMesh.hpp"

#include "DeviceMesh.cuh"

#include "HostScene.hpp"
#include "DeviceScene.cuh"

#include "maths/maths_linear_algebra.cuh"
#include "utility.hpp"

HostMesh::HostMesh(DeviceMesh* device_mesh)
{
	m_deviceMesh = device_mesh;
	name = device_mesh->name;
	Mat4 invmat = m_deviceMesh->inverseModelMatrix;
	modelMatrix = glm::mat4(
		invmat[0].x, invmat[0].y, invmat[0].z, invmat[0].w,
		invmat[1].x, invmat[1].y, invmat[1].z, invmat[1].w,
		invmat[2].x, invmat[2].y, invmat[2].z, invmat[2].w,
		invmat[3].x, invmat[3].y, invmat[3].z, invmat[3].w
	);
	modelMatrix = glm::inverse(modelMatrix);

	triangle_offset_idx = device_mesh->triangle_offset_idx;
	tri_count = device_mesh->tri_count;
}

void HostMesh::updateDevice(HostScene* hscene)
{
	if (m_deviceMesh != nullptr) {
		//print_matrix(modelMatrix);

		glm::mat4 invmat = glm::inverse(modelMatrix);

		Mat4 dinvmat(
			invmat[0][0], invmat[0][1], invmat[0][2], invmat[0][3],  // First column
			invmat[1][0], invmat[1][1], invmat[1][2], invmat[1][3],  // Second column
			invmat[2][0], invmat[2][1], invmat[2][2], invmat[2][3],  // Third column
			invmat[3][0], invmat[3][1], invmat[3][2], invmat[3][3]   // Fourth column
		);

		Mat4 dmodmat = dinvmat.inverse();

		//Mat4::print_matrix(dmodmat);

		hscene->m_DeviceScene->DeviceBLASes[m_deviceMesh->BLAS_idx].setTransform(dmodmat);
		m_deviceMesh->inverseModelMatrix = dinvmat;
	}
}