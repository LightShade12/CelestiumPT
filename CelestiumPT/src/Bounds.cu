#include "bounds.cuh"
#include "maths/linear_algebra.cuh"
#include "ray.cuh"

float Bounds3f::getSurfaceArea() const
{
	float planex = 2 * (pMax.z - pMin.z) * (pMax.y - pMin.y);
	float planey = 2 * (pMax.z - pMin.z) * (pMax.x - pMin.x);
	float planez = 2 * (pMax.x - pMin.x) * (pMax.y - pMin.y);
	return planex + planey + planez;
}

float3 Bounds3f::getCentroid() const
{
	return 0.5f * pMin + 0.5f * pMax;
}

__host__ void Bounds3f::adaptBounds(const Mat4& model_mat, const Bounds3f& origin)
{
	// Original AABB corners before transformation
	float3 corners[8] = {
		origin.pMin,                                 // (min.x, min.y, min.z)
		make_float3(origin.pMax.x, origin.pMin.y, origin.pMin.z),  // (max.x, min.y, min.z)
		make_float3(origin.pMin.x, origin.pMax.y, origin.pMin.z),  // (min.x, max.y, min.z)
		make_float3(origin.pMin.x, origin.pMin.y, origin.pMax.z),  // (min.x, min.y, max.z)
		make_float3(origin.pMax.x, origin.pMax.y, origin.pMin.z),  // (max.x, max.y, min.z)
		make_float3(origin.pMin.x, origin.pMax.y, origin.pMax.z),  // (min.x, max.y, max.z)
		make_float3(origin.pMax.x, origin.pMin.y, origin.pMax.z),  // (max.x, min.y, max.z)
		origin.pMax                                  // (max.x, max.y, max.z)
	};

	// Variables to hold new min and max points
	float3 newMin = make_float3(FLT_MAX);
	float3 newMax = make_float3(-FLT_MAX);

	// Transform all 8 corners and compute new bounds
	for (int i = 0; i < 8; ++i) {
		// Transform the corner by the matrix (assume `transform` is a 4x4 matrix)
		float3 transformedPoint = make_float3(model_mat * make_float4(corners[i], 1.0f));

		// Update the new bounding box
		newMin = fminf(newMin, transformedPoint); // fminf compares each component (x, y, z)
		newMax = fmaxf(newMax, transformedPoint); // fmaxf compares each component (x, y, z)
	}

	// Set the new bounds
	pMin = newMin;
	pMax = newMax;
}

__device__ float Bounds3f::intersect(const Ray& ray) const
{
	float3 t0 = (pMin - ray.getOrigin()) * ray.getInvDirection();
	float3 t1 = (pMax - ray.getOrigin()) * ray.getInvDirection();

	float3 tmin = fminf(t0, t1);
	float3 tmax = fmaxf(t1, t0);//switched order of t to guard NaNs

	//min max componenet
	float tenter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	float texit = fminf(fminf(tmax.x, tmax.y), tmax.z);

	// Adjust tenter if the ray starts inside the AABB
	if (tenter < 0.0f) {
		tenter = 0.0f;
	}

	if (tenter > texit || texit < 0) {
		return -1; // No intersection
	}

	return tenter;
}