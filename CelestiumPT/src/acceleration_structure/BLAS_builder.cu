#include "BLAS_builder.hpp"
#include "BLAS.cuh"
#include "host_scene.hpp"
#include "device_scene.cuh"
#include "device_mesh.cuh"
#include "BVH_cache.cuh"

#include <vector>
#include <algorithm>
#include <execution>

void PreProcess(const thrust::universal_vector<Triangle>& tris, std::vector<BVHPrimitiveBounds>& primbounds) {
	printf("Creating prim bounds cache\n");
	for (int prim_idx = 0; prim_idx < tris.size(); prim_idx++)
	{
		float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
		const Triangle* tri = thrust::raw_pointer_cast(&tris[prim_idx]);
		float3 positions[3] = { tri->vertex0.position, tri->vertex1.position, tri->vertex2.position };
		for (float3 pos : positions)
		{
			min.x = fminf(min.x, pos.x);
			min.y = fminf(min.y, pos.y);
			min.z = fminf(min.z, pos.z);

			max.x = fmaxf(max.x, pos.x);
			max.y = fmaxf(max.y, pos.y);
			max.z = fmaxf(max.z, pos.z);
		}
		primbounds.push_back(BVHPrimitiveBounds(min, max));
	}
	printf("prim bounds cached! \n-------------------------------------\n");
}

void BLASBuilder::build(HostScene* hscene)
{
	DeviceScene* dscene = hscene->m_DeviceScene;
	std::vector<BVHNode>hnodes;
	std::vector<size_t>prim_indices;
	std::vector<BVHPrimitiveBounds>prim_bounds;
	//preprocess
	PreProcess(dscene->DeviceTriangles, prim_bounds);

	const thrust::universal_vector<Triangle>& read_prims = dscene->DeviceTriangles;

	BLAS::BVHBuilderSettings cfg;
	cfg.m_TargetLeafPrimitivesCount = 6;

	for (size_t meshidx = 0; meshidx < dscene->DeviceMeshes.size(); meshidx++) {
		DeviceMesh* dmesh = thrust::raw_pointer_cast(&(dscene->DeviceMeshes[meshidx]));
		std::string name = dmesh->name;
		fprintf(stderr, "%zu> BLAS mesh: %s\n", meshidx, name.c_str());
		BLAS blas(dmesh, meshidx, read_prims, hnodes, prim_indices, prim_bounds, cfg);//local space build

		Mat4 d_inv_mat = dmesh->inverseModelMatrix;

		blas.m_invModelMatrix = dmesh->inverseModelMatrix;
		blas.m_BoundingBox.pMax = make_float3(d_inv_mat.inverse() * make_float4(hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMax, 1));
		blas.m_BoundingBox.pMin = make_float3(d_inv_mat.inverse() * make_float4(hnodes[blas.m_BVHRootIdx].m_BoundingBox.pMin, 1));
		blas.m_Original_bounding_box = blas.m_BoundingBox;

		dscene->DeviceBLASes.push_back(blas);
		dmesh->BLAS_idx = (meshidx);
	}
	dscene->DeviceBVHNodes = hnodes;
	dscene->DeviceBVHTriangleIndices = prim_indices;

	dscene->syncDeviceGeometry();
}