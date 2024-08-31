#include "Integrator.cuh"
#include "Triangle.cuh"
#include "ShapeIntersection.cuh"

#include "device_launch_parameters.h"
#define __CUDACC__
#include <surface_indirect_functions.h>
#include <float.h>

__device__ uint32_t pcg_hash(uint32_t input)
{
	uint32_t state = input * 747796405u + 2891336453u;
	uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
//0-1
__device__ float randF_PCGHash(uint32_t& seed)
{
	seed = pcg_hash(seed);
	return (float)seed / (float)UINT32_MAX;
}

__device__ float get1D_PCGHash(uint32_t& seed) { return randF_PCGHash(seed); };
__device__ float2 get2D_PCGHash(uint32_t& seed) { return make_float2(get1D_PCGHash(seed), get1D_PCGHash(seed)); };
__device__ float2 getPixel2D_PCGHash(uint32_t& seed) { return get2D_PCGHash(seed); };

void IntegratorPipeline::invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims)
{
	renderKernel << < block_grid_dims, thread_block_dims >> > (globals);
};

__global__ void renderKernel(IntegratorGlobals globals)
{
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 frameres = globals.FrameBuffer.resolution;

	if ((thread_pixel_coord_x >= frameres.x) || (thread_pixel_coord_y >= frameres.y)) return;

	float3 sampled_radiance = IntegratorPipeline::evaluatePixelSample(globals, { (float)thread_pixel_coord_x,(float)thread_pixel_coord_y });

	float4 fragcolor = { sampled_radiance.x,sampled_radiance.y,sampled_radiance.z, 1 };

	surf2Dwrite(fragcolor, globals.FrameBuffer.composite_render_surface_object, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}

__device__ float3 IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel)
{
	uint32_t seed = ppixel.x + ppixel.y * globals.FrameBuffer.resolution.x;
	seed *= globals.frameidx;
	int2 frameres = globals.FrameBuffer.resolution;
	float2 screen_uv = { (ppixel.x / frameres.x),(ppixel.y / frameres.y) };
	screen_uv = screen_uv * 2 - 1;
	Ray primary_ray = globals.SceneDescriptor.dev_camera->generateRay(frameres.x, frameres.y, screen_uv);

	float3 L = IntegratorPipeline::Li(globals, primary_ray);

	//return make_float3(screen_uv);
	//return make_float3(get2D_PCGHash(seed), get1D_PCGHash(seed));
	return L;
}
__constant__ const float TRIANGLE_EPSILON = 0.000001;//TODO: place proper

__device__ ShapeIntersection Intersection(const Ray& ray, const Triangle& triangle, int triangle_idx)
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

//return -1 hit_dist
__device__ ShapeIntersection MissStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload) {
	return ShapeIntersection();
}

__device__ ShapeIntersection ClosestHitStage(const IntegratorGlobals& globals, const Ray& ray, const ShapeIntersection& in_payload) {
	const Triangle& triangle = globals.SceneDescriptor.dev_aggregate->DeviceTrianglesBuffer[in_payload.triangle_idx];

	ShapeIntersection out_payload;

	out_payload.bary = in_payload.bary;
	out_payload.triangle_idx = in_payload.triangle_idx;
	out_payload.hit_distance = in_payload.hit_distance;

	out_payload.w_pos = ray.getOrigin() + ray.getDirection() * in_payload.hit_distance;

	if (dot(triangle.face_normal, -1 * ray.getDirection()) < 0.f)
	{
		out_payload.front_face = false;
		out_payload.w_norm = -1.f * triangle.face_normal;
	}
	else {
		out_payload.w_norm = triangle.face_normal;
		out_payload.front_face = true;
	}

	return out_payload;
}

__device__ ShapeIntersection IntegratorPipeline::Intersect(const IntegratorGlobals& globals, const Ray& ray)
{
	ShapeIntersection payload;
	payload.hit_distance = FLT_MAX;

	for (size_t triangle_idx = 0; triangle_idx < globals.SceneDescriptor.dev_aggregate->DeviceTrianglesCount; triangle_idx++)
	{
		const Triangle& tri = globals.SceneDescriptor.dev_aggregate->DeviceTrianglesBuffer[triangle_idx];
		ShapeIntersection eval_payload = Intersection(ray, tri, triangle_idx);
		if (eval_payload.hit_distance < payload.hit_distance) {
			payload = eval_payload;
		}
	}

	if (payload.triangle_idx == -1) {
		return MissStage(globals, ray, payload);
	}

	return ClosestHitStage(globals, ray, payload);
}

//struct Hitpayload {
//	float dist = -1;
//	float3 w_pos{};
//	float3 w_norm{};
//};
//
//__device__ Hitpayload hitsphere(const Ray& ray) {
//	float3 center = { 0,0,-20 };
//	float radius = 5;
//	float3 oc = center - ray.getOrigin();
//	float a = dot(ray.getDirection(), ray.getDirection());
//	float b = -2.0 * dot(ray.getDirection(), oc);
//	float c = dot(oc, oc) - radius * radius;
//	float discriminant = b * b - 4 * a * c;
//
//	Hitpayload payload;
//	if (discriminant < 0) {
//		payload.dist = -1;
//	}
//	else {
//		payload.dist = (-b - sqrtf(discriminant)) / (2.0 * a);
//		payload.w_pos = ray.getOrigin() + (ray.getDirection() * payload.dist);
//		payload.w_norm = normalize(payload.w_pos - center);
//	}
//	return payload;
//}

__device__ float3 ImageInfiniteLight() {};

__device__ float3 SkyShading(const Ray& ray) {
	float3 unit_direction = normalize(ray.getDirection());
	float a = 0.5f * (unit_direction.y + 1.0);
	return (1.0f - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
};

__device__ float3 IntegratorPipeline::Li(const IntegratorGlobals& globals, const Ray& ray)
{
	return IntegratorPipeline::LiRandomWalk(globals, ray);
}

__device__ float3 IntegratorPipeline::LiRandomWalk(const IntegratorGlobals& globals, const Ray& ray)
{
	//Triangle tri;
	//tri.vertex0 = { {0,3.75,-10},{0,0,1} };
	//tri.vertex1 = { {3,-1.0,-10},{0,0,1} };
	//tri.vertex2 = { {-3,-1.0,-10},{0,0,1} };

	//Hitpayload hitpayload = hitsphere(ray);
	ShapeIntersection payload = Intersect(globals, ray);
	//payload.w_norm = { tri.vertex0.normal * payload.bary.x + tri.vertex1.normal * payload.bary.y + tri.vertex2.normal * payload.bary.z };

	//if (hitpayload.dist < 0)
	if (payload.hit_distance < 0)
	{
		//miss
		return SkyShading(ray);
	}
	//hit
	return payload.w_norm;

	return make_float3(1, 0, 0);
}
;