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
	const thrust::universal_vector<Triangle> read_prims = dscene->DeviceTriangles;

	BLAS::BVHBuilderSettings cfg;
	cfg.m_TargetLeafPrimitivesCount = 8;

	for (int meshidx = 0; meshidx < dscene->DeviceMeshes.size(); meshidx++) {
		DeviceMesh* dmesh = thrust::raw_pointer_cast(&(dscene->DeviceMeshes[meshidx]));
		dscene->DeviceBLASes.push_back(BLAS(dmesh, read_prims, hnodes, prim_indices, cfg));
	}

	dscene->DeviceBVHNodes = hnodes;
	dscene->DeviceBVHTriangleIndices = prim_indices;

#ifdef BVHDEBUG
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
		int r = dscene->DeviceBLASes[idx].bvhrootIdx;
		int c = dscene->DeviceBLASes[idx].bvhnodesCount;
		int s = dscene->DeviceBLASes[idx].bvhnodesStartIdx;
		printf("blas %d: root=%d, count=%d, start=%d,\n", idx, r, c, s);
	}
	printf("\n");
#endif // BVHDEBUG

	dscene->syncDeviceGeometry();
}