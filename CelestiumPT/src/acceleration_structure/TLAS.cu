#include "TLAS.cuh"
#include "Ray.cuh"
#include "Storage.cuh"
#include "SceneGeometry.cuh"
#include "ShapeIntersection.cuh"

TLAS::TLAS(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes)
{
	m_BLASCount = read_blases.size();
	build(read_blases, tlasnodes);
	m_TLASRootIdx = tlasnodes.size() - 1;
	m_TLASnodesCount = tlasnodes.size();
	m_BoundingBox = tlasnodes[m_TLASRootIdx].m_BoundingBox;
};

int TLAS::FindBestMatch(int* list, int N, int A, std::vector<TLASNode>& tlasnodes)
{
	float smallest = FLT_MAX;
	int bestB = -1;
	for (int B = 0; B < N; B++) if (B != A)
	{
		float3 bmax = fmaxf(tlasnodes[list[A]].m_BoundingBox.pMax, tlasnodes[list[B]].m_BoundingBox.pMax);
		float3 bmin = fminf(tlasnodes[list[A]].m_BoundingBox.pMin, tlasnodes[list[B]].m_BoundingBox.pMin);
		float3 e = bmax - bmin;
		float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
		if (surfaceArea < smallest) smallest = surfaceArea, bestB = B;
	}
	return bestB;
}
void TLAS::refresh(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes)
{
}
void TLAS::build(const thrust::universal_vector<BLAS>& read_blases, std::vector<TLASNode>& tlasnodes)
{
	//TEST
	assert(read_blases.size() == 2, "[TEST]MORE THAN 2 BLASES");
	TLASNode tlasleaf1;
	tlasleaf1.BLAS_idx = 0;
	tlasleaf1.m_BoundingBox = Bounds3f({ 10,10,10 }, { 10,10,10 });
	tlasnodes.push_back(tlasleaf1);

	TLASNode tlasleaf2;
	tlasleaf2.BLAS_idx = 1;
	tlasleaf2.m_BoundingBox = Bounds3f({ 10,10,10 }, { 10,10,10 });
	tlasnodes.push_back(tlasleaf2);

	TLASNode tlasroot;
	tlasroot.m_BoundingBox = Bounds3f({ 10,10,10 }, { 10,10,10 });
	tlasroot.leftRight = 0 + (1 << 16);
	tlasnodes.push_back(tlasroot);

	m_BLASCount = read_blases.size();
	m_TLASnodesCount = tlasnodes.size();
	m_TLASRootIdx = tlasnodes.size() - 1;
	return;
	//----------

	// assign a TLASleaf node to each BLAS; making work list
	int* nodeIdx = new int[m_BLASCount];
	int nodeIndices = m_BLASCount;

	for (uint i = 0; i < m_BLASCount; i++)
	{
		TLASNode tlasnode;
		tlasnode.m_BoundingBox = read_blases[i].m_BoundingBox;
		tlasnode.BLAS_idx = i;
		tlasnode.leftRight = 0; // makes it a leaf
		tlasnodes.push_back(tlasnode);
	}

	//TODO:make and add root somewhere
	// use agglomerative clustering to build the TLAS
	int A = 0, B = FindBestMatch(nodeIdx, nodeIndices, A, tlasnodes);
	while (nodeIndices > 1)
	{
		int C = FindBestMatch(nodeIdx, nodeIndices, B, tlasnodes);
		if (A == C)
		{
			int nodeIdxA = nodeIdx[A], nodeIdxB = nodeIdx[B];
			TLASNode& nodeA = tlasnodes[nodeIdxA];
			TLASNode& nodeB = tlasnodes[nodeIdxB];
			TLASNode& newNode = tlasnodes[m_BLASCount];
			newNode.leftRight = nodeIdxA + (nodeIdxB << 16);
			newNode.m_BoundingBox.pMin = fminf(nodeA.m_BoundingBox.pMin, nodeB.m_BoundingBox.pMin);
			newNode.m_BoundingBox.pMax = fmaxf(nodeA.m_BoundingBox.pMax, nodeB.m_BoundingBox.pMax);
			nodeIdx[A] = m_BLASCount++;
			nodeIdx[B] = nodeIdx[nodeIndices - 1];
			B = FindBestMatch(nodeIdx, --nodeIndices, A, tlasnodes);
		}
		else A = B, B = C;
	}
	tlasnodes[0] = tlasnodes[nodeIdx[A]];

	delete[] nodeIdx;
}
__device__ void TLAS::intersect(const IntegratorGlobals& globals, const Ray& ray, ShapeIntersection* closest_hitpayload)
{
	SceneGeometry* sceneGeo = globals.SceneDescriptor.dev_aggregate;

	if (m_BLASCount == 0) return;//empty scene;empty TLAS

	//if (m_BoundingBox.intersect(ray) < 0)return;
	//int nodeIdxA = leftRight & 0xFFFF;        // Extract lower 16 bits (nodeIdxA)
	//int nodeIdxB = (leftRight >> 16) & 0xFFFF; // Extract upper 16 bits (nodeIdxB)

	const uint8_t maxStackSize = 64;//TODO: rename this var
	int nodeIdxStack[maxStackSize];
	float nodeHitDistStack[maxStackSize];
	uint8_t stackPtr = 0;

	float current_node_hitdist = FLT_MAX;

	const TLASNode* stackTopNode = &(sceneGeo->DeviceTLASNodesBuffer[m_TLASRootIdx]);//is this in register?
	nodeIdxStack[stackPtr] = m_TLASRootIdx;
	nodeHitDistStack[stackPtr++] = stackTopNode->m_BoundingBox.intersect(ray);

	//TODO: make the shapeIntersetion shorter
	ShapeIntersection workinghitpayload;//only to be written to by primitive proccessing
	float child1_hitdist = -1;
	float child2_hitdist = -1;

	while (stackPtr > 0) {
		stackTopNode = &(sceneGeo->DeviceTLASNodesBuffer[nodeIdxStack[--stackPtr]]);
		current_node_hitdist = nodeHitDistStack[stackPtr];

		//custom ray interval culling
		//if (!(ray.interval.surrounds(current_node_hitdist)))continue;//TODO: can put this in triangle looping part to get inner clipping working

		//skip nodes farther than closest triangle; redundant: see the ordered traversal code
		if (closest_hitpayload->triangle_idx != -1 && closest_hitpayload->hit_distance < current_node_hitdist)continue;
		if (current_node_hitdist < 0) continue;
		//closest_hitpayload->color += make_float3(1) * 0.05f;

		//if interior
		if (!stackTopNode->isleaf())
		{
			int c1id = stackTopNode->leftRight & 0xFFFF, c2id = (stackTopNode->leftRight >> 16) & 0xFFFF;

			child1_hitdist = (sceneGeo->DeviceTLASNodesBuffer[c1id]).m_BoundingBox.intersect(ray);
			child2_hitdist = (sceneGeo->DeviceTLASNodesBuffer[c2id]).m_BoundingBox.intersect(ray);
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
			sceneGeo->DeviceBLASesBuffer[stackTopNode->BLAS_idx].intersect(globals, ray, closest_hitpayload);
		}
	}
};