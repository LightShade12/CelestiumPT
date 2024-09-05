#include "BLASBuilder.hpp"
#include "BLAS.cuh"
#include "HostScene.hpp"
#include "DeviceScene.cuh"
#include "Mesh.cuh"

void BLASBuilder::build(HostScene* hscene)
{
	DeviceScene* dscene = hscene->m_DeviceScene;

	for (int meshidx = 0; meshidx < dscene->DeviceMeshes.size(); meshidx++) {
		Mesh* dmesh = thrust::raw_pointer_cast(&(dscene->DeviceMeshes[meshidx]));
		dscene->DeviceBLASes.push_back(BLAS(dmesh, dscene, {}));
	}

	dscene->syncDeviceGeometry();
}