#include "host_mesh.hpp"

#include "device_mesh.cuh"

#include "host_scene.hpp"
#include "device_scene.cuh"

#include "maths/linear_algebra.cuh"
#include "utility.hpp"

HostMesh::HostMesh(DeviceMesh* device_mesh)
{
	m_deviceMesh = device_mesh;
	name = device_mesh->name;

	modelMatrix = device_mesh->modelMatrix.toGLM();
	invModelMatrix = device_mesh->inverseModelMatrix.toGLM();

	triangle_offset_idx = device_mesh->triangle_offset_idx;
	tri_count = device_mesh->tri_count;
}

//TODO: resolve two way link better
void HostMesh::updateDevice(HostScene* hscene)
{
	if (m_deviceMesh != nullptr) {
		//print_matrix(modelMatrix);
		Mat4 dmodel = Mat4(modelMatrix);
		Mat4 dinvmodel = Mat4(invModelMatrix);

		//Mat4::print_matrix(dmodmat);
		m_deviceMesh->prev_modelMatrix = m_deviceMesh->modelMatrix;

		hscene->m_DeviceScene->DeviceBLASes[m_deviceMesh->BLAS_idx].setTransform(dmodel);
		m_deviceMesh->inverseModelMatrix = dinvmodel;
		m_deviceMesh->modelMatrix = dmodel;
	}
}