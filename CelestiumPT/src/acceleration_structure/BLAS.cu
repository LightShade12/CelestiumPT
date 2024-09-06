#include "BLAS.cuh"
#include "Storage.cuh"
#include "RayStages.cuh"
#include "Integrator.cuh"
#include "ShapeIntersection.cuh"
#include "DeviceScene.cuh"
#include "Ray.cuh"
#include "Mesh.cuh"
#include "BVHNode.cuh"

#include <float.h>
#include <algorithm>
#include <numeric>

BLAS::BLAS(Mesh* mesh, DeviceScene* dscene, BVHBuilderSettings buildercfg)
{
	MeshLink = mesh;
	uint32_t prim_start_idx = mesh->triangle_offset_idx;
	uint32_t prim_end_idx = prim_start_idx + mesh->tri_count;
	std::vector<BVHNode> bvhnodes;
	std::vector<uint32_t>bvhprim_indices;

	bvhnodesStartIdx = dscene->DeviceBVHNodes.size();
	build(dscene->DeviceTriangles,
		prim_start_idx,
		prim_end_idx,
		dscene->DeviceBVHNodes,
		bvhprim_indices,
		dscene->DeviceBVHTriangleIndices.size(),
		buildercfg);
	bvhnodesCount = dscene->DeviceBVHNodes.size() - bvhnodesStartIdx;
	bvhrootIdx = bvhnodesCount - 1;

	//thrust::universal_vector<BVHNode>nodes = bvhnodes;
	thrust::universal_vector<uint32_t>indices = bvhprim_indices;
	//dscene->DeviceBVHNodes.insert(dscene->DeviceBVHNodes.end(), nodes.begin(), nodes.end());//insert
	dscene->DeviceBVHTriangleIndices.insert(dscene->DeviceBVHTriangleIndices.end(), indices.begin(), indices.end());//insert
};

void BLAS::build(const thrust::universal_vector<Triangle>& read_tris, size_t prim_start_idx,
	size_t prim_end_idx, thrust::universal_vector<BVHNode>& bvhnodes, std::vector<uint32_t>& fresh_primindices, size_t actual_indices_offset, BVHBuilderSettings cfg)
{
	BVHNode* hostBVHroot = new BVHNode();
	//empty scene
	if (read_tris.size() == 0) {
		hostBVHroot->triangle_indices_count = 0;
		hostBVHroot->m_BoundingBox = Bounds3f(make_float3(0, 0, 0), make_float3(0, 0, 0));
		hostBVHroot->left_child_or_triangle_indices_start_idx = -1;
		bvhnodes.push_back(*hostBVHroot); //optional?
		delete hostBVHroot;
		return;
	}

	fresh_primindices.resize(prim_end_idx - prim_start_idx, 0);

	std::iota(fresh_primindices.begin(), fresh_primindices.end(), prim_start_idx);

	float3 minextent;
	float3 extent = get_Absolute_Extent(read_tris, fresh_primindices,
		0, fresh_primindices.size(),
		minextent);

	hostBVHroot->m_BoundingBox = Bounds3f(minextent, minextent + extent);
	hostBVHroot->left_child_or_triangle_indices_start_idx = 0;//if leaf; will be valid
	hostBVHroot->triangle_indices_count = fresh_primindices.size();
	printf("root prim count:%d \n", hostBVHroot->triangle_indices_count);

	// If root leaf candidate
	if (hostBVHroot->triangle_indices_count <= cfg.m_TargetLeafPrimitivesCount)
	{
		bvhnodes.push_back(*hostBVHroot);
		delete hostBVHroot;
		printf("<<!! -- made ROOT leaf with %d prims -- !!>>\n", hostBVHroot->triangle_indices_count);

		return;
	}

	const int MAX_STACK_SIZE = 512; // Adjust this value as needed as per expected depth
	BVHNode* nodesToBeBuilt[MAX_STACK_SIZE]{};//max postponed nodes //TODO:make this node indices to avoid host_bvh_nodes preallocation limitation
	int stackPtr = 0;

	size_t nodecount = 1024 * 100;
	bvhnodes.reserve(nodecount);

	// Static stack for iterative BVH construction
	nodesToBeBuilt[stackPtr++] = hostBVHroot;
	BVHNode* currentNode = nullptr;

	while (stackPtr > 0)
	{
		currentNode = nodesToBeBuilt[--stackPtr];

		// If the current node is a leaf candidate
		if (currentNode->triangle_indices_count <= cfg.m_TargetLeafPrimitivesCount)
		{
			printf(">>> made a leaf node with %d prims --------------->\n", currentNode->triangle_indices_count);
			currentNode->left_child_or_triangle_indices_start_idx += actual_indices_offset;
			continue;
		}

		// Partition the current node
		BVHNode* leftNode = new BVHNode();
		BVHNode* rightNode = new BVHNode();

		//retval nodes always have triangle indices
		makePartition(read_tris, fresh_primindices,
			currentNode->left_child_or_triangle_indices_start_idx, currentNode->left_child_or_triangle_indices_start_idx + currentNode->triangle_indices_count,//as leaf
			*leftNode, *rightNode, cfg);
		currentNode->triangle_indices_count = 0;//mark as not leaf

		printf("size before child1pushback: %zu\n", bvhnodes.size());
		bvhnodes.push_back(*leftNode); delete leftNode;
		currentNode->left_child_or_triangle_indices_start_idx = bvhnodes.size() - 1;//as interior node
		printf("size after child1pushback: %zu\n", bvhnodes.size());

		bvhnodes.push_back(*rightNode); delete rightNode;
		printf("size after child2pushback: %zu\n", bvhnodes.size());

		printf("child1 idx %d\n", currentNode->left_child_or_triangle_indices_start_idx);
		printf("child2 idx %d\n", currentNode->left_child_or_triangle_indices_start_idx + 1);

		// Push the child nodes onto the stack
		nodesToBeBuilt[stackPtr++] = &bvhnodes[currentNode->left_child_or_triangle_indices_start_idx];
		nodesToBeBuilt[stackPtr++] = &bvhnodes[currentNode->left_child_or_triangle_indices_start_idx + 1];
	}

	bvhnodes.push_back(*hostBVHroot); delete hostBVHroot;
	bvhnodes.shrink_to_fit();

	return;
}

int BLAS::costHeursitic(const BVHNode& left_node, const BVHNode& right_node, const Bounds3f& parent_bbox, BVHBuilderSettings bvhcfg)
{
	return bvhcfg.m_RayAABBIntersectionCost +
		((left_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * left_node.triangle_indices_count * bvhcfg.m_RayPrimitiveIntersectionCost) +
		((right_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * right_node.triangle_indices_count * bvhcfg.m_RayPrimitiveIntersectionCost);
}

float3 BLAS::get_Absolute_Extent(const std::vector<const Triangle*>& primitives, size_t start_idx, size_t end_idx, float3& min_extent)
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

float3 BLAS::get_Absolute_Extent(const thrust::universal_vector<Triangle>& primitives_, const std::vector<unsigned int>& primitive_indices,
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

void BLAS::makePartition(const thrust::universal_vector<Triangle>& read_tris, std::vector<uint32_t>& primitives_indices,
	size_t start_idx, size_t end_idx, BVHNode& leftnode, BVHNode& rightnode, BVHBuilderSettings cfg)
{
	printf("---> making partition, input prim count:%zu <---\n", end_idx - start_idx);
	float lowest_cost_partition_pt = 0;//best bin
	PartitionAxis best_partition_axis{};

	int lowest_cost = INT_MAX;

	float3 minextent = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 extent = get_Absolute_Extent(read_tris, primitives_indices,
		start_idx, end_idx, minextent);//TODO:can be replaced with caller node's bounds
	Bounds3f parent_bbox(minextent, minextent + extent);

	BVHNode temp_left_node, temp_right_node;

	//for x
	std::vector<float>bins;//world space
	bins.reserve(cfg.binCount);
	float deltapartition = extent.x / cfg.binCount;
	for (int i = 1; i < cfg.binCount; i++)
	{
		bins.push_back(minextent.x + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc x bin %.3f\n", bin);
		binToShallowNodes(temp_left_node, temp_right_node,
			bin, PartitionAxis::X_AXIS,
			read_tris, primitives_indices, start_idx, end_idx);
		/*int cost = BVHNode::trav_cost + ((temp_left_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * temp_left_node.primitives_count * temp_left_node.rayint_cost) +
			((temp_right_node.getSurfaceArea() / parent_bbox.getSurfaceArea()) * temp_right_node.primitives_count * temp_right_node.rayint_cost);*/
		int cost = costHeursitic(temp_left_node, temp_right_node, parent_bbox, cfg);
		if (cost < lowest_cost)
		{
			lowest_cost = cost;
			best_partition_axis = PartitionAxis::X_AXIS;
			lowest_cost_partition_pt = bin;
		}
	}

	//for y
	bins.clear();
	deltapartition = extent.y / cfg.binCount;
	for (int i = 1; i < cfg.binCount; i++)
	{
		bins.push_back(minextent.y + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc y bin %.3f\n", bin);
		binToShallowNodes(temp_left_node, temp_right_node, bin, PartitionAxis::Y_AXIS,
			read_tris, primitives_indices, start_idx, end_idx);
		int cost = costHeursitic(temp_left_node, temp_right_node, parent_bbox, cfg);
		if (cost < lowest_cost)
		{
			lowest_cost = cost;
			best_partition_axis = PartitionAxis::Y_AXIS;
			lowest_cost_partition_pt = bin;
		}
	}

	//for z
	bins.clear();
	deltapartition = extent.z / cfg.binCount;
	for (int i = 1; i < cfg.binCount; i++)
	{
		bins.push_back(minextent.z + (i * deltapartition));
	}
	for (float bin : bins)
	{
		//printf("proc z bin %.3f\n", bin);
		binToShallowNodes(temp_left_node, temp_right_node, bin, PartitionAxis::Z_AXIS,
			read_tris, primitives_indices, start_idx, end_idx);
		int cost = costHeursitic(temp_left_node, temp_right_node, parent_bbox, cfg);
		if (cost < lowest_cost)
		{
			lowest_cost = cost;
			best_partition_axis = PartitionAxis::Z_AXIS;
			lowest_cost_partition_pt = bin;
		}
	}

	printf(">> made a partition, bin: %.3f, axis: %d, cost: %d <<---\n", lowest_cost_partition_pt, best_partition_axis, lowest_cost);
	binToNodes(leftnode, rightnode, lowest_cost_partition_pt,
		best_partition_axis, read_tris, primitives_indices, start_idx, end_idx);
	printf("left node prim count:%d | right node prim count: %d\n", leftnode.triangle_indices_count, rightnode.triangle_indices_count);
}

void BLAS::binToNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
	const thrust::universal_vector<Triangle>& read_tris, std::vector<uint32_t>& primitives_indices, size_t start_idx, size_t end_idx)
{
	//sorting

	std::vector<unsigned int>::iterator partition_iterator;
	switch (axis)
	{
	case PartitionAxis::X_AXIS:
		partition_iterator = std::partition(primitives_indices.begin() + start_idx, primitives_indices.begin() + end_idx,
			[bin, read_tris](unsigned int prim_idx) { return read_tris[prim_idx].centroid.x < bin; });
		break;
	case PartitionAxis::Y_AXIS:
		partition_iterator = std::partition(primitives_indices.begin() + start_idx, primitives_indices.begin() + end_idx,
			[bin, read_tris](unsigned int prim_idx) { return read_tris[prim_idx].centroid.y < bin; });
		break;
	case PartitionAxis::Z_AXIS:
		partition_iterator = std::partition(primitives_indices.begin() + start_idx, primitives_indices.begin() + end_idx,
			[bin, read_tris](unsigned int prim_idx) { return read_tris[prim_idx].centroid.z < bin; });
		break;
	default:
		break;
	}

	int partition_start_idx = std::distance(primitives_indices.begin(), partition_iterator);
	left.left_child_or_triangle_indices_start_idx = start_idx;//as leaf
	left.triangle_indices_count = partition_start_idx - start_idx;
	float3 leftminextent;
	float3 leftextent = get_Absolute_Extent(read_tris, primitives_indices, left.left_child_or_triangle_indices_start_idx,
		left.left_child_or_triangle_indices_start_idx + left.triangle_indices_count, leftminextent);
	left.m_BoundingBox = Bounds3f(leftminextent, leftminextent + leftextent);

	right.left_child_or_triangle_indices_start_idx = partition_start_idx;
	right.triangle_indices_count = end_idx - partition_start_idx;
	float3 rightminextent;
	float3 rightextent = get_Absolute_Extent(read_tris, primitives_indices, right.left_child_or_triangle_indices_start_idx,
		right.left_child_or_triangle_indices_start_idx + right.triangle_indices_count, rightminextent);
	right.m_BoundingBox = Bounds3f(rightminextent, rightminextent + rightextent);
}

void BLAS::binToShallowNodes(BVHNode& left, BVHNode& right, float bin, PartitionAxis axis,
	const thrust::universal_vector<Triangle>& read_tris, std::vector<uint32_t>& primitives_indices, size_t start_idx, size_t end_idx)
{
	//can make a single vector and run partition
	std::vector<const Triangle*>left_prim_ptrs;
	std::vector<const Triangle*>right_prim_ptrs;

	for (size_t prim_indice_idx = start_idx; prim_indice_idx < end_idx; prim_indice_idx++)
	{
		const Triangle* triangle = &(read_tris[primitives_indices[prim_indice_idx]]);

		switch (axis)
		{
		case PartitionAxis::X_AXIS:
			if (triangle->centroid.x < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		case PartitionAxis::Y_AXIS:
			if (triangle->centroid.y < bin)left_prim_ptrs.push_back(triangle);
			else right_prim_ptrs.push_back(triangle);
			break;
		case PartitionAxis::Z_AXIS:
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

//-----------------------------------------------------------------------------------------------------------------------
__device__ void BLAS::intersect(const IntegratorGlobals& globals, const Ray& ray, ShapeIntersection* closest_hitpayload)
{
	SceneGeometry* sceneGeo = globals.SceneDescriptor.dev_aggregate;
	if (bvhnodesCount == 0) return;//empty scene

	const uint8_t maxStackSize = 64;
	int nodeIdxStack[maxStackSize];
	float nodeHitDistStack[maxStackSize];
	uint8_t stackPtr = 0;

	float current_node_hitdist = FLT_MAX;

	nodeIdxStack[stackPtr] = bvhrootIdx;
	const BVHNode* stackTopNode = &(sceneGeo->DeviceBVHNodesBuffer[bvhrootIdx]);//is this in register?
	nodeHitDistStack[stackPtr++] = stackTopNode->m_BoundingBox.intersect(ray);

	//TODO: make the shaprIntersetion shorter
	ShapeIntersection workinghitpayload;//only to be written to by primitive proccessing
	float child1_hitdist = -1;
	float child2_hitdist = -1;
	const Triangle* primitive = nullptr;

	while (stackPtr > 0) {
		stackTopNode = &(sceneGeo->DeviceBVHNodesBuffer[nodeIdxStack[--stackPtr]]);
		current_node_hitdist = nodeHitDistStack[stackPtr];

		//custom ray interval culling
		//if (!(ray.interval.surrounds(current_node_hitdist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		//skip nodes farther than closest triangle; redundant
		if (closest_hitpayload->triangle_idx != -1 && closest_hitpayload->hit_distance < current_node_hitdist)continue;

		//closest_hitpayload->color += make_float3(1) * 0.05f;

		//if interior
		if (stackTopNode->triangle_indices_count <= 0)
		{
			child1_hitdist = (sceneGeo->DeviceBVHNodesBuffer[stackTopNode->left_child_or_triangle_indices_start_idx]).m_BoundingBox.intersect(ray);
			child2_hitdist = (sceneGeo->DeviceBVHNodesBuffer[stackTopNode->left_child_or_triangle_indices_start_idx + 1]).m_BoundingBox.intersect(ray);
			//TODO:implement early cull properly see discord for ref
			if (child1_hitdist > child2_hitdist) {
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_child_or_triangle_indices_start_idx; }
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_child_or_triangle_indices_start_idx + 1; }
			}
			else {
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_child_or_triangle_indices_start_idx + 1; }
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance) { nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = stackTopNode->left_child_or_triangle_indices_start_idx; }
			}
		}
		else
		{
			for (int primIndiceIdx = stackTopNode->left_child_or_triangle_indices_start_idx;
				primIndiceIdx < stackTopNode->left_child_or_triangle_indices_start_idx + stackTopNode->triangle_indices_count; primIndiceIdx++) {
				int primIdx = sceneGeo->DeviceBVHTriangleIndicesBuffer[primIndiceIdx];
				primitive = &(sceneGeo->DeviceTrianglesBuffer[primIdx]);
				workinghitpayload = IntersectionStage(ray, *primitive, primIdx);

				if (workinghitpayload.triangle_idx != -1 && workinghitpayload.hit_distance < closest_hitpayload->hit_distance) {
					//if (!AnyHit(ray, sceneGeo, &workinghitpayload))continue;
					closest_hitpayload->hit_distance = workinghitpayload.hit_distance;
					closest_hitpayload->triangle_idx = workinghitpayload.triangle_idx;
					closest_hitpayload->bary = workinghitpayload.bary;
				}
			}
		}
	}
}