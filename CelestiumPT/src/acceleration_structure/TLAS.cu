#include "TLAS.cuh"
#include "ray.cuh"
#include "storage.cuh"
#include "scene_geometry.cuh"
#include "shape_intersection.cuh"

TLAS::TLAS(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes)
{
	m_BLASCount = read_blases.size();
	build(read_blases, tlasnodes);
	m_TLASRootIdx = int(tlasnodes.size() - 1);
	m_TLASnodesCount = tlasnodes.size();
	m_BoundingBox = tlasnodes[m_TLASRootIdx].m_BoundingBox;
};

int TLAS::FindBestMatch(int* TLASwork_idx_list, int work_idx_list_size, int TLAS_A_idx, std::vector<TLASNode>& tlasnodes)
{
	float smallest = FLT_MAX;
	int bestB = -1;
	for (int B = 0; B < work_idx_list_size; B++) if (B != TLAS_A_idx)
	{
		float3 bmax = fmaxf(tlasnodes[TLASwork_idx_list[TLAS_A_idx]].m_BoundingBox.pMax, tlasnodes[TLASwork_idx_list[B]].m_BoundingBox.pMax);
		float3 bmin = fminf(tlasnodes[TLASwork_idx_list[TLAS_A_idx]].m_BoundingBox.pMin, tlasnodes[TLASwork_idx_list[B]].m_BoundingBox.pMin);
		float3 e = bmax - bmin;
		float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;//half SA
		if (surfaceArea < smallest) smallest = surfaceArea, bestB = B;
	}
	return bestB;
}
void TLAS::refresh(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes)
{
	m_BLASCount = read_blases.size();
	build(read_blases, tlasnodes);
	m_TLASRootIdx = tlasnodes.size() - 1;
	m_TLASnodesCount = tlasnodes.size();
	m_BoundingBox = tlasnodes[m_TLASRootIdx].m_BoundingBox;
}
void TLAS::build(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes)
{
	int* TLASnodeIdx = new int[m_BLASCount];
	int nodeIndices = m_BLASCount;//work list size
	// assign a TLASleaf node to each BLAS; making work list
	for (size_t i = 0; i < m_BLASCount; i++)
	{
		TLASnodeIdx[i] = tlasnodes.size();//i derived from m_BLASCount also works
		TLASNode tlasnode;
		tlasnode.m_BoundingBox = read_blases[i].m_BoundingBox;
		tlasnode.BLAS_idx = i;
		tlasnode.leftRight = 0; // makes it a leaf
		tlasnodes.push_back(tlasnode);
	}

	// use agglomerative clustering to build the TLAS
	int A = 0, B = FindBestMatch(TLASnodeIdx, nodeIndices, A, tlasnodes);
	while (nodeIndices > 1)
	{
		int C = FindBestMatch(TLASnodeIdx, nodeIndices, B, tlasnodes);
		if (A == C)
		{
			int nodeIdxA = TLASnodeIdx[A], nodeIdxB = TLASnodeIdx[B];
			const TLASNode* nodeA = &(tlasnodes[nodeIdxA]);
			const TLASNode* nodeB = &(tlasnodes[nodeIdxB]);
			TLASNode newNode;
			newNode.leftRight = nodeIdxA + (nodeIdxB << 16);
			newNode.m_BoundingBox.pMin = fminf(nodeA->m_BoundingBox.pMin, nodeB->m_BoundingBox.pMin);
			newNode.m_BoundingBox.pMax = fmaxf(nodeA->m_BoundingBox.pMax, nodeB->m_BoundingBox.pMax);
			tlasnodes.push_back(newNode);
			TLASnodeIdx[A] = (tlasnodes.size() - 1);
			TLASnodeIdx[B] = TLASnodeIdx[nodeIndices - 1];
			B = FindBestMatch(TLASnodeIdx, --nodeIndices, A, tlasnodes);
		}
		else A = B, B = C;
	}
	//TODO:make and add root somewhere
	//tlasnodes[0] = tlasnodes[TLASnodeIdx[A]];
	tlasnodes.push_back(tlasnodes[TLASnodeIdx[A]]);

	delete[] TLASnodeIdx;
}

#define TLAS_TRAVERSAL_MAX_STACK_DEPTH 16

__device__ void TLAS::intersect(const IntegratorGlobals& globals, const Ray& ray, ShapeIntersection* closest_hitpayload) const
{
	const SceneGeometry* scene_data = globals.SceneDescriptor.DeviceGeometryAggregate;

	if (m_BLASCount == 0) return;//empty scene;empty TLAS

	//if (m_BoundingBox.intersect(ray) < 0)return;

	int nodeIdxStack[TLAS_TRAVERSAL_MAX_STACK_DEPTH];
	float nodeHitDistStack[TLAS_TRAVERSAL_MAX_STACK_DEPTH];
	uint8_t stackPtr = 0;

	float current_node_hitdist = FLT_MAX;

	const TLASNode* stackTopNode = &(scene_data->DeviceTLASNodesBuffer[m_TLASRootIdx]);//is this in register?
	nodeIdxStack[stackPtr] = m_TLASRootIdx;
	nodeHitDistStack[stackPtr++] = stackTopNode->m_BoundingBox.intersect(ray);

	float child1_hitdist = -1;
	float child2_hitdist = -1;

	while (stackPtr > 0) {
		stackTopNode = &(scene_data->DeviceTLASNodesBuffer[nodeIdxStack[--stackPtr]]);
		current_node_hitdist = nodeHitDistStack[stackPtr];

		//custom ray interval culling
		//if (!(ray.interval.surrounds(current_node_hitdist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		if (current_node_hitdist >= 0)closest_hitpayload->GAS_debug += make_float3(0, 0, 1) * globals.IntegratorCFG.GAS_shading_brightness;
		//skip nodes farther than closest triangle; redundant: see the ordered traversal code
		if (closest_hitpayload->hasHit() && closest_hitpayload->hit_distance < current_node_hitdist)continue;

		//if interior
		if (!stackTopNode->isleaf())
		{
			int c1id = stackTopNode->leftRight & 0xFFFF, c2id = (stackTopNode->leftRight >> 16) & 0xFFFF;

			child1_hitdist = (scene_data->DeviceTLASNodesBuffer[c1id]).m_BoundingBox.intersect(ray);
			child2_hitdist = (scene_data->DeviceTLASNodesBuffer[c2id]).m_BoundingBox.intersect(ray);
			//TODO:implement early cull properly see discord for ref
			if (child1_hitdist > child2_hitdist) {
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance)
				{
					nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = c1id;
				}
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance)
				{
					nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = c2id;
				}
			}
			else {
				if (child2_hitdist >= 0 && child2_hitdist < closest_hitpayload->hit_distance)
				{
					nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = c2id;
				}
				if (child1_hitdist >= 0 && child1_hitdist < closest_hitpayload->hit_distance)
				{
					nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = c1id;
				}
			}
		}
		else//if leaf
		{
			scene_data->DeviceBLASesBuffer[stackTopNode->BLAS_idx].intersect(globals, ray, closest_hitpayload);
		}
	}
}

//TODO: optimise P methods via culling
__device__ bool TLAS::intersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax) const
{
	const SceneGeometry* scene_data = globals.SceneDescriptor.DeviceGeometryAggregate;

	if (m_BLASCount == 0) return;//empty scene;empty TLAS

	//if (m_BoundingBox.intersect(ray) < 0)return;

	int nodeIdxStack[TLAS_TRAVERSAL_MAX_STACK_DEPTH];
	float nodeHitDistStack[TLAS_TRAVERSAL_MAX_STACK_DEPTH];
	uint8_t stackPtr = 0;

	const TLASNode* stackTopNode = &(scene_data->DeviceTLASNodesBuffer[m_TLASRootIdx]);//is this in register?
	nodeIdxStack[stackPtr] = m_TLASRootIdx;
	nodeHitDistStack[stackPtr++] = stackTopNode->m_BoundingBox.intersect(ray);

	float child1_hitdist = -1;
	float child2_hitdist = -1;

	bool hit = false;

	while (stackPtr > 0 && !hit) {
		stackTopNode = &(scene_data->DeviceTLASNodesBuffer[nodeIdxStack[--stackPtr]]);

		//custom ray interval culling
		//if (!(ray.interval.surrounds(current_node_hitdist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		//if interior
		if (!stackTopNode->isleaf())
		{
			int c1id = stackTopNode->leftRight & 0xFFFF, c2id = (stackTopNode->leftRight >> 16) & 0xFFFF;

			child1_hitdist = (scene_data->DeviceTLASNodesBuffer[c1id]).m_BoundingBox.intersect(ray);
			child2_hitdist = (scene_data->DeviceTLASNodesBuffer[c2id]).m_BoundingBox.intersect(ray);
			//TODO:implement early cull properly see discord for ref
			if (child1_hitdist > child2_hitdist) {
				if (child1_hitdist >= 0 && child1_hitdist < tmax)
				{
					nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = c1id;
				}
				if (child2_hitdist >= 0 && child2_hitdist < tmax)
				{
					nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = c2id;
				}
			}
			else {
				if (child2_hitdist >= 0 && child2_hitdist < tmax)
				{
					nodeHitDistStack[stackPtr] = child2_hitdist; nodeIdxStack[stackPtr++] = c2id;
				}
				if (child1_hitdist >= 0 && child1_hitdist < tmax)
				{
					nodeHitDistStack[stackPtr] = child1_hitdist; nodeIdxStack[stackPtr++] = c1id;
				}
			}
		}
		else//if leaf
		{
			hit |= scene_data->DeviceBLASesBuffer[stackTopNode->BLAS_idx].intersectP(globals, ray, tmax);
		}
	}
	return hit;
}
;