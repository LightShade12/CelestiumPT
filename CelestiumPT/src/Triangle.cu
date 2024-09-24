#include "Triangle.cuh"

//TODO:lookup docs
__device__ float3 sampleUniformTriangle(float2 u)
{
	float2 bary;
	if (u.x < u.y) {
		bary.x = u.x / 2;
		bary.y = u.y - bary.x;
	}
	else {
		bary.y = u.y / 2;
		bary.x = u.x - bary.y;
	}
	return { bary.x, bary.y, 1 - bary.x - bary.y };
}

Triangle::Triangle(Vertex v0, Vertex v1, Vertex v2, glm::vec3 nrm) :vertex0(v0), vertex1(v1), vertex2(v2),
face_normal(make_float3(nrm.x, nrm.y, nrm.z))
{
	centroid = (vertex0.position + vertex1.position + vertex2.position) / 3;
};

__host__ __device__ float Triangle::area()
{
	float3 edge0 = vertex1.position - vertex0.position;
	float3 edge1 = vertex2.position - vertex0.position;
	return 0.5f * length(cross(edge0, edge1));
}

__device__ ShapeSample Triangle::sample(const ShapeSampleContext& ctx, float2 u2)
{
	float3 p0 = vertex0.position, p1 = vertex1.position, p2 = vertex2.position;
	float3 bary = sampleUniformTriangle(u2);
	float3 p = p0 * bary.x + p1 * bary.y + p2 * bary.z;
	return ShapeSample(p, face_normal, 1.f / area());
};