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
		BLAS blas(dmesh, read_prims, hnodes, prim_indices, cfg);

		Mat4 mat = dmesh->modelMatrix;
		glm::mat4 gmodelmat = glm::mat4(
			mat[0].x, mat[0].y, mat[0].z, mat[0].w,
			mat[1].x, mat[1].y, mat[1].z, mat[1].w,
			mat[2].x, mat[2].y, mat[2].z, mat[2].w,
			mat[3].x, mat[3].y, mat[3].z, mat[3].w
		);
		glm::vec3 pmin = glm::vec3(
			hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMin.x,
			hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMin.y,
			hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMin.z);
		glm::vec3 pmax = glm::vec3(
			hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMax.x,
			hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMax.y,
			hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMax.z);
		pmin = gmodelmat * glm::vec4(pmin, 1);
		pmax = gmodelmat * glm::vec4(pmax, 1);
		blas.m_BoundingBox = Bounds3f(make_float3(pmin.x, pmin.y, pmin.z), make_float3(pmax.x, pmax.y, pmax.z));
		dscene->DeviceBLASes.push_back(blas);
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
		int r = dscene->DeviceBLASes[idx].m_BVHRootIdx;
		int c = dscene->DeviceBLASes[idx].m_BVHNodesCount;
		int s = dscene->DeviceBLASes[idx].m_BVHNodesStartIdx;
		printf("blas %d: root=%d, count=%d, start=%d,\n", idx, r, c, s);
	}
	printf("\n");
#endif // BVHDEBUG

	dscene->syncDeviceGeometry();
}