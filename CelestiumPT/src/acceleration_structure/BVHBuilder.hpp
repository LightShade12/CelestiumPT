#pragma once
//#include "HostScene.hpp"
//#include "BVHNode.cuh"
#include <memory>
#include <vector>
struct BVHNode;
struct Bounds3f;
class HostScene;

//TODO: spatial splits for 20% more improvement
class BVHBuilder
{
public:
	BVHBuilder() = default;
	BVHNode* BuildIterative(HostScene* scene);
	BVHNode* BuildBVH(HostScene& scene);

private:
	enum class PartitionAxis
	{
		X_AXIS = 0,
		Y_AXIS,
		Z_AXIS
	};

	void recursiveBuild(BVHNode& node, std::vector<BVHNode>& bvh_nodes,
		const  HostScene& scene, std::vector<unsigned int>& primitive_indices);

	//bin is in world space
	void binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const  HostScene& scene, std::vector<unsigned int>& primitives_indices, size_t start_idx, size_t end_idx);
	void binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
		const  HostScene& scene, std::vector<unsigned int>& primitives_indices, size_t start_idx, size_t end_idx);
	int costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox);
	void makePartition(const  HostScene& scene, std::vector<unsigned int>& primitives_indices,
		size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode);

public:
	int m_BinCount = 32;
	int m_TargetLeafPrimitivesCount = 6;
	int m_RayPrimitiveIntersectionCost = 2;
	int m_RayAABBIntersectionCost = 1;
};