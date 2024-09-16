#include "BLASBuilder.hpp"
#include "BLAS.cuh"
#include "HostScene.hpp"
#include "DeviceScene.cuh"
#include "DeviceMesh.cuh"
#include <vector>

void BLASBuilder::build(HostScene* hscene)
{
	DeviceScene* dscene = hscene->m_DeviceScene;
	std::vector<BVHNode>hnodes;
	std::vector<int>prim_indices;
	const thrust::universal_vector<Triangle>& read_prims = dscene->DeviceTriangles;

	BLAS::BVHBuilderSettings cfg;
	cfg.m_TargetLeafPrimitivesCount = 6;

	for (int meshidx = 0; meshidx < dscene->DeviceMeshes.size(); meshidx++) {
		DeviceMesh* dmesh = thrust::raw_pointer_cast(&(dscene->DeviceMeshes[meshidx]));
		BLAS blas(dmesh, read_prims, hnodes, prim_indices, cfg);//local space build

		Mat4 d_inv_mat = dmesh->inverseModelMatrix;

		blas.invModelMatrix = dmesh->inverseModelMatrix;
		blas.m_BoundingBox.pMax = d_inv_mat.inverse() * make_float4(hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMax, 1);
		blas.m_BoundingBox.pMin = d_inv_mat.inverse() * make_float4(hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMin, 1);
		blas.m_Original_bounding_box = blas.m_BoundingBox;

		dscene->DeviceBLASes.push_back(blas);
		dmesh->BLAS_idx = (dscene->DeviceBLASes.size() - 1);
	}
	dscene->DeviceBVHNodes = hnodes;
	dscene->DeviceBVHTriangleIndices = prim_indices;

	dscene->syncDeviceGeometry();
}