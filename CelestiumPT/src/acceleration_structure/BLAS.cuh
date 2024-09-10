#pragma once

#include "Triangle.cuh"
#include "Bounds.cuh"

//#include <vector_types.h>
#include <cuda_runtime.h>
#include <thrust/universal_vector.h>

#include <vector>
#include <cstdint>

class DeviceScene;
struct IntegratorGlobals;
class DeviceMesh;
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
		uint32_t m_BinCount = 32;
	};

	BLAS() = default;
	BLAS(DeviceMesh* mesh, const thrust::universal_vector<Triangle>& prims, std::vector<BVHNode>& nodes,
		std::vector<int>& prim_indices, BVHBuilderSettings buildercfg);

	__device__ void intersect(const IntegratorGlobals& globals, const Ray& ray, ShapeIntersection* closest_hitpayload);

private:
	void build(const thrust::universal_vector<Triangle>& read_tris, size_t prim_start_idx, size_t prim_end_idx,
		std::vector<BVHNode>& bvhnodes, std::vector<int>& prim_indices, BVHBuilderSettings cfg);

	int costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox, BVHBuilderSettings bvhcfg);

	//reads prim_indices from pos=start_idx to end_idx to access and compute triangles bound
	float3 get_Absolute_Extent(const thrust::universal_vector<Triangle>& primitives_, const std::vector<int>& primitive_indices,
		size_t start_idx_, size_t end_idx_, float3& min_extent_);

	//retval nodes always have triangle indices assigned
	void makePartition(const thrust::universal_vector<Triangle>& read_tris, std::vector<int>& primitives_indices,
		size_t start_idx, size_t end_idx, BVHNode* leftnode, BVHNode* rightnode, BVHBuilderSettings cfg);

	//this overload is used for making temp shallow bin partition nodes
	float3 get_Absolute_Extent(const std::vector<const Triangle*>& primitives, size_t start_idx, size_t end_idx, float3& min_extent);

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const thrust::universal_vector<Triangle>& read_tris, std::vector<int>& primitives_indices, size_t start_idx, size_t end_idx);

	void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const thrust::universal_vector<Triangle>& read_tris, const std::vector<int>& primitives_indices, size_t start_idx, size_t end_idx);

public:

	Bounds3f m_BoundingBox;//Must be in World space
	uint32_t m_BVHNodesCount = 0;
	int m_BVHNodesStartIdx = -1;
	int m_BVHRootIdx = -1;//Ideally should be startidx too
	DeviceMesh* m_MeshLink = nullptr;
};