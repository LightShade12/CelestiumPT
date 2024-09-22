#include "RayStages.cuh"
#include "Ray.cuh"
#include "Storage.cuh"
#include "SceneGeometry.cuh"
#include "ShapeIntersection.cuh"
#include "Triangle.cuh"
#include "maths/constants.cuh"

__device__ ShapeIntersection MissStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload) {
	ShapeIntersection out_payload;
	out_payload.GAS_debug = in_payload.GAS_debug;
	return out_payload;
}

__device__ ShapeIntersection ClosestHitStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload)
{
	const Triangle& triangle = globals.SceneDescriptor.device_geometry_aggregate->DeviceTrianglesBuffer[in_payload.triangle_idx];

	ShapeIntersection out_payload;

	out_payload.bary = in_payload.bary;
	out_payload.triangle_idx = in_payload.triangle_idx;
	out_payload.hit_distance = in_payload.hit_distance;
	out_payload.GAS_debug = in_payload.GAS_debug;
	out_payload.invModelMatrix = in_payload.invModelMatrix;
	out_payload.arealight = in_payload.arealight;

	Mat4 model_matrix = in_payload.invModelMatrix.inverse();
	out_payload.w_pos = ray.getOrigin() + (ray.getDirection() * in_payload.hit_distance);

	out_payload.w_geo_norm = triangle.face_normal;

	out_payload.uv =
		(triangle.vertex0.UV * out_payload.bary.x) +
		(triangle.vertex1.UV * out_payload.bary.y) +
		(triangle.vertex2.UV * out_payload.bary.z);

	out_payload.w_shading_norm = normalize(
		(triangle.vertex0.normal * out_payload.bary.x) +
		(triangle.vertex1.normal * out_payload.bary.y) +
		(triangle.vertex2.normal * out_payload.bary.z)
	);

	if (dot(
		normalize(make_float3(model_matrix * make_float4(out_payload.w_geo_norm, 0))),
		-1 * ray.getDirection()) < 0.f)
	{
		out_payload.front_face = false;
		out_payload.w_geo_norm = normalize(make_float3(model_matrix * make_float4(-1.f * out_payload.w_geo_norm, 0)));
		out_payload.w_shading_norm = normalize(make_float3(model_matrix * make_float4(-1.f * out_payload.w_shading_norm, 0)));
	}
	else {
		out_payload.w_geo_norm = normalize(make_float3(model_matrix * make_float4(out_payload.w_geo_norm, 0)));
		out_payload.w_shading_norm = normalize(make_float3(model_matrix * make_float4(out_payload.w_shading_norm, 0)));
		out_payload.front_face = true;
	}

	return out_payload;
}

__device__ void IntersectionStage(const Ray& ray, const Triangle& triangle, int triangle_idx, CompactShapeIntersection* payload)
{
	payload->hit_distance = FLT_MAX;
	payload->triangle_idx = -1;

	float3 v0v1 = triangle.vertex1.position - triangle.vertex0.position;
	float3 v0v2 = triangle.vertex2.position - triangle.vertex0.position;

	float3 pvec = cross(ray.getDirection(), v0v2);

	float det = dot(v0v1, pvec);
	if (det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON)
		return; //payload; // This ray is parallel to this triangle

	float invDet = 1.0f / det;
	float3 tvec = ray.getOrigin() - triangle.vertex0.position;
	float u = invDet * dot(tvec, pvec);
	if (u < 0.0f || u > 1.0f)
		return;// payload;

	float3 qvec = cross(tvec, v0v1);
	float v = invDet * dot(ray.getDirection(), qvec);
	if (v < 0.0f || u + v > 1.0f)
		return;// payload;

	float t = invDet * dot(v0v2, qvec);
	if (t > TRIANGLE_EPSILON) { // ray intersection
		payload->hit_distance = t;
		payload->triangle_idx = triangle_idx;
		payload->bary = { 1.0f - u - v, u, v };
		return; //payload;
	}

	return;// payload;
};