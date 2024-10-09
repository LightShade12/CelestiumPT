#include "triangle.cuh"

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

Triangle::Triangle(Vertex v0, Vertex v1, Vertex v2, glm::vec3 nrm, int matidx) :vertex0(v0), vertex1(v1), vertex2(v2),
face_normal(make_float3(nrm.x, nrm.y, nrm.z)), mat_idx(matidx)
{
	centroid = (vertex0.position + vertex1.position + vertex2.position) / 3;
}
//TODO: complete this
__device__ float Triangle::PDF(const ShapeSampleContext& ctx, float3 wi) const
{
	//float dist = length(ctx.p - ls.pLight);
	//float dist_sq = dist * dist;
	//float cosTheta_emitter = AbsDot(wi, ls.n);
	//float Li_sample_pdf = (sampled_light.p * ls.pdf) * (1 / cosTheta_emitter) * dist_sq;
	return 0;
}
;

__host__ __device__ float Triangle::area() const
{
	float3 edge0 = vertex1.position - vertex0.position;
	float3 edge1 = vertex2.position - vertex0.position;
	return 0.5f * length(cross(edge0, edge1));
}

__device__ ShapeSample Triangle::sample(const ShapeSampleContext& ctx, float2 u2) const
{
	float3 p0 = vertex0.position, p1 = vertex1.position, p2 = vertex2.position;
	float3 bary = sampleUniformTriangle(u2);
	float3 p = p0 * bary.x + p1 * bary.y + p2 * bary.z;
	return ShapeSample(p, face_normal, 1.f / area());
};