#pragma once
#include "Triangle.cuh"

#include <thrust/universal_vector.h>
#include <cuda_runtime.h>

class DeviceScene;
struct Bounds3f;
struct IntegratorGlobals;
class Mesh;
struct BVHNode;
class Ray;
struct ShapeIntersection;

class BLAS {
public:
	enum class PartitionAxis
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	struct BVHBuilderSettings {
		BVHBuilderSettings() = default;
		int m_RayAABBIntersectionCost = 1;
		int m_RayPrimitiveIntersectionCost = 2;
		uint32_t m_TargetLeafPrimitivesCount = 8;
		uint32_t binCount = 32;
	};

	BLAS() = default;
	BLAS(Mesh* mesh, DeviceScene* dscene, BVHBuilderSettings buildercfg);

	void build(const thrust::universal_vector<Triangle>& read_tris, size_t prim_start_idx, size_t prim_end_idx,
		thrust::universal_vector<BVHNode>& bvhnodes, std::vector<uint32_t>& fresh_primindices, size_t actual_indices_offset, BVHBuilderSettings cfg);

	int costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox, BVHBuilderSettings bvhcfg);

	//reads prim_indices from pos=start_idx to end_idx to access and compute triangles bound
	float3 get_Absolute_Extent(const thrust::universal_vector<Triangle>& primitives_, const std::vector<unsigned int>& primitive_indices,
		size_t start_idx_, size_t end_idx_, float3& min_extent_);

	//reads prim_indices from pos=start_idx to end_idx to access and compute triangles bound
	float3 get_Absolute_Extent(const std::vector<const Triangle*>& primitives, size_t start_idx, size_t end_idx, float3& min_extent);

	//retval nodes always have triangle indices assigned
	void makePartition(const thrust::universal_vector<Triangle>& read_tris, std::vector<uint32_t>& primitives_indices,
		size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode, BVHBuilderSettings cfg);

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const thrust::universal_vector<Triangle>& read_tris, std::vector<uint32_t>& primitives_indices, size_t start_idx, size_t end_idx);

	void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const thrust::universal_vector<Triangle>& read_tris, std::vector<uint32_t>& primitives_indices, size_t start_idx, size_t end_idx);

	__device__ void intersect(const IntegratorGlobals& globals, const Ray& ray, ShapeIntersection* closest_hitpayload);
public:
	uint32_t bvhnodesCount = 0;
	int bvhnodesStartIdx = -1;
	int bvhrootIdx = -1;//Ideally should be startidx too
	Mesh* MeshLink = nullptr;
};