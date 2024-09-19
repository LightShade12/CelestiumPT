#include "BVHCache.cuh"

BVHPrimitiveCache::BVHPrimitiveCache(Triangle triangle)
{
	float3 min = { FLT_MAX,FLT_MAX,FLT_MAX }, max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	float3 positions[3] = { triangle.vertex0.position, triangle.vertex1.position, triangle.vertex2.position };
	for (float3 pos : positions)
	{
		min.x = fminf(min.x, pos.x);
		min.y = fminf(min.y, pos.y);
		min.z = fminf(min.z, pos.z);

		max.x = fmaxf(max.x, pos.x);
		max.y = fmaxf(max.y, pos.y);
		max.z = fmaxf(max.z, pos.z);
	}
	centroid = (positions[0] + positions[1] + positions[2]) / 3.f;
}