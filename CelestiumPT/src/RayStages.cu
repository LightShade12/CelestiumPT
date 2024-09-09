#include "RayStages.cuh"
#include "Ray.cuh"
#include "Storage.cuh"
#include "SceneGeometry.cuh"
#include "ShapeIntersection.cuh"
#include "Triangle.cuh"
#include "maths/constants.cuh"

__device__ ShapeIntersection MissStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload) {
	return ShapeIntersection();
}

__device__ ShapeIntersection ClosestHitStage(const IntegratorGlobals& globals, const Ray& ray, const Mat4& model_matrix, const ShapeIntersection& in_payload)
{
	const Triangle& triangle = globals.SceneDescriptor.dev_aggregate->DeviceTrianglesBuffer[in_payload.triangle_idx];

	ShapeIntersection out_payload;

	out_payload.bary = in_payload.bary;
	out_payload.triangle_idx = in_payload.triangle_idx;
	out_payload.hit_distance = in_payload.hit_distance;

	out_payload.w_pos = model_matrix.transpose() * (ray.getOrigin() + ray.getDirection() * in_payload.hit_distance);//TODO:problem part
	out_payload.w_pos -= model_matrix.transpose() * model_matrix[3];

	//TODO: implement smooth shading here
	if (dot(triangle.face_normal, -1 * ray.getDirection()) < 0.f)
	{
		out_payload.front_face = false;
		out_payload.w_norm = normalize(model_matrix.transpose() * (-1.f * triangle.face_normal));
	}
	else {
		out_payload.w_norm = normalize(model_matrix.transpose() * triangle.face_normal);
		out_payload.front_face = true;
	}

	return out_payload;
}

__device__ ShapeIntersection IntersectionStage(const Ray& ray, const Triangle& triangle, int triangle_idx)
{
	ShapeIntersection payload;

	float3 v0v1 = triangle.vertex1.position - triangle.vertex0.position;
	float3 v0v2 = triangle.vertex2.position - triangle.vertex0.position;

	float3 pvec = cross(ray.getDirection(), v0v2);
	float det = dot(v0v1, pvec);
	if (det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON)
		return payload; // This ray is parallel to this triangle

	float invDet = 1.0f / det;
	float3 tvec = ray.getOrigin() - triangle.vertex0.position;
	float u = invDet * dot(tvec, pvec);
	if (u < 0.0f || u > 1.0f)
		return payload;

	float3 qvec = cross(tvec, v0v1);
	float v = invDet * dot(ray.getDirection(), qvec);
	if (v < 0.0f || u + v > 1.0f)
		return payload;

	float t = invDet * dot(v0v2, qvec);
	if (t > TRIANGLE_EPSILON) { // ray intersection
		payload.hit_distance = t;
		payload.triangle_idx = triangle_idx;
		payload.bary = { 1.0f - u - v, u, v };
		return payload;
	}

	return payload;
};