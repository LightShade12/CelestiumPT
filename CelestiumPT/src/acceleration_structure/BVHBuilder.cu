#include "BVHBuilder.hpp"

#include "Triangle.cuh"
#include "HostScene.hpp"
#include "DeviceScene.cuh"

//#include "Editor/Common/dbg_macros.hpp"

#include "maths/vector_maths.cuh"
#include <stack>
#include <thrust/universal_vector.h>
#include <thrust/partition.h>
#include <algorithm>
#include <numeric>

//TODO: change to getExtents; implicitly handles a vector of triangle in build code
	//bounding box extents
float3 get_Absolute_Extent(const thrust::universal_vector<Triangle>& primitives_, const std::vector<unsigned int>& primitive_indices,
	size_t start_idx_, size_t end_idx_, float3& min_extent_)
{
	float3 extent;
	float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

	for (int prim_indice_idx = start_idx_; prim_indice_idx < end_idx_; prim_indice_idx++)
	{
		const Triangle* tri = &(primitives_[primitive_indices[prim_indice_idx]]);
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
	}
	min_extent_ = min;
	extent = { max.x - min.x,max.y - min.y,max.z - min.z };
	return extent;
};

float3 get_Absolute_Extent(const std::vector<const Triangle*>& primitives, size_t start_idx, size_t end_idx, float3& min_extent)
{
	float3 extent;
	float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

	for (int prim_idx = start_idx; prim_idx < end_idx; prim_idx++)
	{
		const Triangle* tri = (primitives[prim_idx]);
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
	}
	min_extent = min;
	extent = { max.x - min.x,max.y - min.y,max.z - min.z };
	return extent;
};

// The build function initializes the root and uses a stack for iterative processing

BVHNode* BVHBuilder::BuildIterative(HostScene* scene)
{
	BVHNode* hostBVHroot = new BVHNode();

	if (scene->getTrianglesCount() == 0) {
		hostBVHroot->triangle_indices_count = 0;
		hostBVHroot->m_BoundingBox = Bounds3f(make_float3(0, 0, 0), make_float3(0, 0, 0));
		hostBVHroot->left_child_or_triangle_indices_start_idx = -1;
		//scene->m_DeviceScene->DeviceBVHNodes.push_back(*hostBVHroot); //optional
		delete hostBVHroot;
		return nullptr;
	}
	//int MAX_STACK_SIZE = 2 * ((primitives.size() + primitives.size() / m_TargetLeafPrimitivesCount) / 2) - 1; // Adjust this value as needed
	std::vector <unsigned int> host_prim_indices(scene->getTrianglesCount());
	std::iota(host_prim_indices.begin(), host_prim_indices.end(), 0);

	float3 minextent;
	float3 extent = get_Absolute_Extent(scene->m_DeviceScene->DeviceTriangles, host_prim_indices, 0, host_prim_indices.size(),
		minextent);

	hostBVHroot->m_BoundingBox = Bounds3f(minextent, minextent + extent);
	hostBVHroot->left_child_or_triangle_indices_start_idx = 0;//as leaf
	hostBVHroot->triangle_indices_count = host_prim_indices.size();
	printf("root prim count:%zu \n", hostBVHroot->triangle_indices_count);

	const int MAX_STACK_SIZE = 512; // Adjust this value as needed as per expected depth
	BVHNode* nodesToBeBuilt[MAX_STACK_SIZE]{};//max postponed nodes //TODO:make this node indices to avoid host_bvh_nodes preallocation limitation
	int stackPtr = 0;

	//primitive_indices.resize(primitives.size());

	std::vector <BVHNode> host_bvh_nodes;
	size_t nodecount = 1024 * 100;
	host_bvh_nodes.reserve(nodecount);

	// If leaf candidate
	if (hostBVHroot->triangle_indices_count <= m_TargetLeafPrimitivesCount)
	{
		//hostBVHroot->m_IsLeaf = true;

		scene->m_DeviceScene->DeviceBVHNodes.push_back(*hostBVHroot);
		delete hostBVHroot;
		printf("<<!! -- made ROOT leaf with %d prims -- !!>>\n", hostBVHroot->triangle_indices_count);

		//return thrust::raw_pointer_cast(&(bvh_nodes.back()));
		return nullptr;//TODO: tie up loose ends
	}

	// Static stack for iterative BVH construction
	nodesToBeBuilt[stackPtr++] = hostBVHroot;
	BVHNode* currentNode = nullptr;

	while (stackPtr > 0)
	{
		currentNode = nodesToBeBuilt[--stackPtr];

		// If the current node is a leaf candidate
		if (currentNode->triangle_indices_count <= m_TargetLeafPrimitivesCount)
		{
			printf(">>> made a leaf node with %d prims --------------->\n", currentNode->triangle_indices_count);
			//currentNode->m_IsLeaf = true;
			continue;
		}

		// Partition the current node
		BVHNode* leftNode = new BVHNode();
		BVHNode* rightNode = new BVHNode();

		makePartition(*scene, host_prim_indices,
			currentNode->left_child_or_triangle_indices_start_idx, currentNode->left_child_or_triangle_indices_start_idx + currentNode->triangle_indices_count,//as leaf
			*leftNode, *rightNode);
		currentNode->triangle_indices_count = 0;//mark as not leaf

		printf("size before child1pushback: %zu\n", host_bvh_nodes.size());
		host_bvh_nodes.push_back(*leftNode); delete leftNode;
		currentNode->left_child_or_triangle_indices_start_idx = host_bvh_nodes.size() - 1;//as node
		printf("size after child1pushback: %zu\n", host_bvh_nodes.size());

		host_bvh_nodes.push_back(*rightNode); delete rightNode;
		//currentNode->dev_child2_idx = host_bvh_nodes.size() - 1;
		printf("size after child2pushback: %zu\n", host_bvh_nodes.size());

		printf("child1 idx %d\n", currentNode->left_child_or_triangle_indices_start_idx);
		printf("child2 idx %d\n", currentNode->left_child_or_triangle_indices_start_idx + 1);

		// Push the child nodes onto the stack
		nodesToBeBuilt[stackPtr++] = &host_bvh_nodes[currentNode->left_child_or_triangle_indices_start_idx];
		nodesToBeBuilt[stackPtr++] = &host_bvh_nodes[currentNode->left_child_or_triangle_indices_start_idx + 1];
	}

	host_bvh_nodes.push_back(*hostBVHroot); delete hostBVHroot;
	host_bvh_nodes.shrink_to_fit();

	scene->m_DeviceScene->DeviceBVHTriangleIndices = host_prim_indices;

	scene->m_DeviceScene->DeviceBVHNodes = host_bvh_nodes;

	//return thrust::raw_pointer_cast(&(bvh_nodes.back()));
	return nullptr;//TODO: fix loose ends
}

/*
TODO: sort triangle scene array into contiguous bvh leaf tri groups
Union of bbox in SAH bin computing
*/

//TODO:copy tris to host for bvh, let tris be dev_vector
BVHNode* BVHBuilder::BuildBVH(HostScene& scene)
{
	std::shared_ptr<BVHNode>host_BVH_root = std::make_shared<BVHNode>();

	std::vector<BVHNode>host_bvh_nodes;
	printf("<--bvh build input tris count:%zu -->\n", scene.getTrianglesCount());

	std::vector <unsigned int> host_prim_indices(scene.getTrianglesCount());
	std::iota(host_prim_indices.begin(), host_prim_indices.end(), 0);

	float3 minextent;
	float3 extent = get_Absolute_Extent(scene.m_DeviceScene->DeviceTriangles, host_prim_indices,
		0, scene.m_DeviceScene->DeviceTriangles.size(), minextent);
	host_BVH_root->m_BoundingBox = Bounds3f(minextent, minextent + extent);

	//if leaf candidate
	if (scene.getTrianglesCount() <= m_TargetLeafPrimitivesCount)
	{
		host_BVH_root->triangle_indices_count = scene.getTrianglesCount();
		//host_BVH_root->m_IsLeaf = true;
		host_BVH_root->left_child_or_triangle_indices_start_idx = 0;//as leaf

		scene.m_DeviceScene->DeviceBVHNodes.push_back(*host_BVH_root);
		printf("-----made RootNode leaf with %d prims-----\n", host_BVH_root->triangle_indices_count);

		//return thrust::raw_pointer_cast(&(bvh_nodes.back()));
		return nullptr;
	}

	std::shared_ptr<BVHNode>left = std::make_shared<BVHNode>();
	std::shared_ptr<BVHNode>right = std::make_shared<BVHNode>();

	makePartition(scene, host_prim_indices,
		0, scene.m_DeviceScene->DeviceTriangles.size(), *left, *right);

	recursiveBuild(*left, host_bvh_nodes, scene, host_prim_indices);
	recursiveBuild(*right, host_bvh_nodes, scene, host_prim_indices);

	host_bvh_nodes.push_back(*left);
	host_BVH_root->left_child_or_triangle_indices_start_idx = scene.m_DeviceScene->DeviceBVHNodes.size() - 1;//as node

	host_bvh_nodes.push_back(*right);
	//host_BVH_root->dev_child2_idx = bvh_nodes.size() - 1;

	scene.m_DeviceScene->DeviceBVHTriangleIndices = host_prim_indices;
	host_bvh_nodes.push_back(*host_BVH_root);
	scene.m_DeviceScene->DeviceBVHNodes = host_bvh_nodes;
	//return thrust::raw_pointer_cast(&(bvh_nodes.back()));
	return nullptr;
}

void BVHBuilder::recursiveBuild(BVHNode& node, std::vector<BVHNode>& bvh_nodes,
	const  HostScene& scene, std::vector<unsigned int>& primitive_indices)
{
	printf("> recursive child build,input prim count: %d \n", node.triangle_indices_count);

	if (node.triangle_indices_count <= m_TargetLeafPrimitivesCount)
	{
		//mark as leaf
		printf("made a leaf node with %d prims---------------<\n", node.triangle_indices_count);
		//node.left_child_or_triangle_indices_start_idx = -1;// , node.dev_child2_idx = -1;//redundant?
		//node.m_IsLeaf = true; return;
	}
	else
	{
		std::shared_ptr<BVHNode>left_node = std::make_shared<BVHNode>();//TODO: candidate for raw ptr
		std::shared_ptr<BVHNode>right_node = std::make_shared<BVHNode>();

		makePartition(scene, primitive_indices,
			node.left_child_or_triangle_indices_start_idx, node.left_child_or_triangle_indices_start_idx + node.triangle_indices_count,//as leaf
			*left_node, *right_node);
		node.triangle_indices_count = 0;//mark as not leaf

		recursiveBuild(*left_node, bvh_nodes, scene, primitive_indices);
		recursiveBuild(*right_node, bvh_nodes, scene, primitive_indices);

		bvh_nodes.push_back(*left_node);
		node.left_child_or_triangle_indices_start_idx = bvh_nodes.size() - 1;

		bvh_nodes.push_back(*right_node);
		//node.dev_child2_idx = bvh_nodes.size() - 1;
	}
}

//left_child_or_triangle_indices_start_idx will be overwritten to nodes from the caller
__host__ void BVHBuilder::binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
	const  HostScene& scene, std::vector<unsigned int>& primitives_indices, size_t start_idx, size_t end_idx)
{
	//sorting

	std::vector<unsigned int>::iterator partition_iterator;
	switch (axis)
	{
	case BVHBuilder::PartitionAxis::X_AXIS:
		partition_iterator = std::partition(primitives_indices.begin() + start_idx, primitives_indices.begin() + end_idx,
			[bin, scene](unsigned int prim_idx) { return scene.m_DeviceScene->DeviceTriangles[prim_idx].centroid.x < bin; });
		break;
	case BVHBuilder::PartitionAxis::Y_AXIS:
		partition_iterator = std::partition(primitives_indices.begin() + start_idx, primitives_indices.begin() + end_idx,
			[bin, scene](unsigned int prim_idx) { return scene.m_DeviceScene->DeviceTriangles[prim_idx].centroid.y < bin; });
		break;
	case BVHBuilder::PartitionAxis::Z_AXIS:
		partition_iterator = std::partition(primitives_indices.begin() + start_idx, primitives_indices.begin() + end_idx,
			[bin, scene](unsigned int prim_idx) { return scene.m_DeviceScene->DeviceTriangles[prim_idx].centroid.z < bin; });
		break;
	default:
		break;
	}

	int partition_start_idx = std::distance(primitives_indices.begin(), partition_iterator);
	left.left_child_or_triangle_indices_start_idx = start_idx;//as leaf
	left.triangle_indices_count = partition_start_idx - start_idx;
	float3 leftminextent;
	float3 leftextent = get_Absolute_Extent(scene.m_DeviceScene->DeviceTriangles, primitives_indices, left.left_child_or_triangle_indices_start_idx,
		left.left_child_or_triangle_indices_start_idx + left.triangle_indices_count, leftminextent);
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.left_child_or_triangle_indices_start_idx = partition_start_idx;
	right.triangle_indices_count = end_idx - partition_start_idx;
	float3 rightminextent;
	float3 rightextent = get_Absolute_Extent(scene.m_DeviceScene->DeviceTriangles, primitives_indices, right.left_child_or_triangle_indices_start_idx,
		right.left_child_or_triangle_indices_start_idx + right.triangle_indices_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

void BVHBuilder::binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis, const  HostScene& scene,
	std::vector<unsigned int>& primitives_indices, size_t start_idx, size_t end_idx)
{
	//can make a single vector and run partition
	std::vector<const Triangle*>left_prim_ptrs;
	std::vector<const Triangle*>right_prim_ptrs;

	for (size_t prim_indice_idx = start_idx; prim_indice_idx < end_idx; prim_indice_idx++)
	{
		const Triangle* triangle = &(scene.m_DeviceScene->DeviceTriangles[primitives_indices[prim_indice_idx]]);

		switch (axis)
		{
		case BVHBuilder::PartitionAxis::X_AXIS:
			if (triangle->centroid.x < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		case BVHBuilder::PartitionAxis::Y_AXIS:
			if (triangle->centroid.y < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		case BVHBuilder::PartitionAxis::Z_AXIS:
			if (triangle->centroid.z < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		default:
			break;
		}
	}

	left.triangle_indices_count = left_prim_ptrs.size();
	float3 leftminextent;
	float3 leftextent = get_Absolute_Extent(left_prim_ptrs, 0, left_prim_ptrs.size(), leftminextent);
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.triangle_indices_count = right_prim_ptrs.size();
	float3 rightminextent;
	float3 rightextent = get_Absolute_Extent(right_prim_ptrs, 0, right.triangle_indices_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

int BVHBuilder::costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox) {
	return m_RayAABBIntersectionCost +
		((left_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * left_node.triangle_indices_count * m_RayPrimitiveIntersectionCost) +
		((right_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * right_node.triangle_indices_count * m_RayPrimitiveIntersectionCost);
}

void BVHBuilder::makePartition(const HostScene& scene, std::vector<unsigned int>& primitives_indices,
	size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode)
{
	printf("---> making partition, input prim count:%zu <---\n", end_idx - start_idx);
	float lowest_cost_partition_pt = 0;//best bin
	PartitionAxis best_partition_axis{};

	int lowest_cost = INT_MAX;

	float3 minextent = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 extent = get_Absolute_Extent(scene.m_DeviceScene->DeviceTriangles, primitives_indices, start_idx, end_idx, minextent);//TODO:can be replaced with caller node's bounds
	Bounds3f parent_bbox(minextent, minextent + extent);

	BVHNode temp_left_node, temp_right_node;

	//for x
	std::vector<float>bins;//world space
	bins.reserve(m_BinCount);
	float deltapartition = extent.x / m_BinCount;
	for (int i = 1; i < m_BinCount; i++)
	{
		bins.push_back(minextent.x + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc x bin %.3f\n", bin);
		binToShallowNodes(temp_left_node, temp_right_node,
			bin, PartitionAxis::X_AXIS,
			scene, primitives_indices, start_idx, end_idx);
		/*int cost = BVHNode::trav_cost + ((temp_left_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * temp_left_node.primitives_count * temp_left_node.rayint_cost) +
			((temp_right_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * temp_right_node.primitives_count * temp_right_node.rayint_cost);*/
		int cost = costHeursitic(temp_left_node, temp_right_node, parent_bbox);
		if (cost < lowest_cost)
		{
			lowest_cost = cost;
			best_partition_axis = PartitionAxis::X_AXIS;
			lowest_cost_partition_pt = bin;
		}
	}

	//for y
	bins.clear();
	deltapartition = extent.y / m_BinCount;
	for (int i = 1; i < m_BinCount; i++)
	{
		bins.push_back(minextent.y + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc y bin %.3f\n", bin);
		binToShallowNodes(temp_left_node, temp_right_node, bin, PartitionAxis::Y_AXIS,
			scene, primitives_indices, start_idx, end_idx);
		int cost = costHeursitic(temp_left_node, temp_right_node, parent_bbox);
		if (cost < lowest_cost)
		{
			lowest_cost = cost;
			best_partition_axis = PartitionAxis::Y_AXIS;
			lowest_cost_partition_pt = bin;
		}
	}

	//for z
	bins.clear();
	deltapartition = extent.z / m_BinCount;
	for (int i = 1; i < m_BinCount; i++)
	{
		bins.push_back(minextent.z + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc z bin %.3f\n", bin);
		binToShallowNodes(temp_left_node, temp_right_node, bin, PartitionAxis::Z_AXIS,
			scene, primitives_indices, start_idx, end_idx);
		int cost = costHeursitic(temp_left_node, temp_right_node, parent_bbox);
		if (cost < lowest_cost)
		{
			lowest_cost = cost;
			best_partition_axis = PartitionAxis::Z_AXIS;
			lowest_cost_partition_pt = bin;
		}
	}

	printf(">> made a partition, bin: %.3f, axis: %d, cost: %d <<---\n", lowest_cost_partition_pt, best_partition_axis, lowest_cost);
	binToNodes(leftnode, rightnode, lowest_cost_partition_pt,
		best_partition_axis, scene, primitives_indices, start_idx, end_idx);
	printf("left node prim count:%d | right node prim count: %d\n", leftnode.triangle_indices_count, rightnode.triangle_indices_count);
}