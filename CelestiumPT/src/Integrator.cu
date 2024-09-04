#include "Integrator.cuh"
#include "Triangle.cuh"
#include "Mesh.cuh"
#include "ShapeIntersection.cuh"
#include "BSDF.cuh"
#include "IntersectionStage.cuh"
#include "acceleration_structure/BVHTraversal.cuh"

#include "maths/constants.cuh"

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

	if (globals.IntegratorCFG.accumulate) {
		globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] += sampled_radiance;
		sampled_radiance = globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] / (globals.frameidx);
	}

	float4 fragcolor = { sampled_radiance.x,sampled_radiance.y,sampled_radiance.z, 1 };

	//EOTF
	fragcolor = make_float4(sqrtf(sampled_radiance.x), sqrtf(sampled_radiance.y), sqrtf(sampled_radiance.z), 1);

	surf2Dwrite(fragcolor, globals.FrameBuffer.composite_render_surface_object, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}

__device__ float3 IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel)
{
	uint32_t seed = ppixel.x + ppixel.y * globals.FrameBuffer.resolution.x;
	seed *= globals.frameidx;

	int2 frameres = globals.FrameBuffer.resolution;

	float2 screen_uv = { (ppixel.x / frameres.x),(ppixel.y / frameres.y) };
	screen_uv = screen_uv * 2 - 1;//-1->1

	Ray primary_ray = globals.SceneDescriptor.dev_camera->generateRay(frameres.x, frameres.y, screen_uv);

	float3 L = IntegratorPipeline::Li(globals, primary_ray, seed);

	//return make_float3(screen_uv);
	//return make_float3(get2D_PCGHash(seed), get1D_PCGHash(seed));
	return L;
}

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

	traverseBVH(ray, globals.SceneDescriptor.dev_aggregate->DeviceBVHNodesCount - 1, &payload,
		globals.SceneDescriptor.dev_aggregate);
	//for (int meshidx = 0; meshidx < globals.SceneDescriptor.dev_aggregate->DeviceMeshesCount; meshidx++) {
	//	Mesh mesh = globals.SceneDescriptor.dev_aggregate->DeviceMeshesBuffer[meshidx];
	//	Mat4 modelMatrix = mesh.modelMatrix;//TODO: figure oyt how to transform rays
	//
	//	for (size_t triangle_idx = mesh.triangle_offset_idx; triangle_idx < mesh.triangle_offset_idx + mesh.tri_count; triangle_idx++)
	//	{
	//		const Triangle& tri = globals.SceneDescriptor.dev_aggregate->DeviceTrianglesBuffer[triangle_idx];
	//		ShapeIntersection eval_payload = IntersectionStage(ray, tri, triangle_idx);
	//		if (eval_payload.hit_distance < payload.hit_distance && eval_payload.triangle_idx>-1) {
	//			payload = eval_payload;
	//		}
	//	}
	//}

	if (payload.triangle_idx == -1) {
		return MissStage(globals, ray, payload);
	}

	return ClosestHitStage(globals, ray, payload);
}

__device__ float3 SkyShading(const Ray& ray) {
	float3 unit_direction = normalize(ray.getDirection());
	float a = 0.5f * (unit_direction.y + 1.0);
	//return make_float3(0.2f, 0.3f, 0.4f);
	return (1.0f - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
};

__device__ float3 IntegratorPipeline::Li(const IntegratorGlobals& globals, const Ray& ray, uint32_t seed)
{
	return IntegratorPipeline::LiRandomWalk(globals, ray, seed);
}

//TODO: replace with tangent space version
__device__ float3 sampleCosineWeightedHemisphere(const float3& normal, float2 xi) {
	// Generate a cosine-weighted direction in the local frame
	float phi = 2.0f * PI * xi.x;
	float cosTheta = sqrtf(xi.y);//TODO: might have to switch with sinTheta
	float sinTheta = sqrtf(1.0f - xi.y);

	float3 H;
	H.x = sinTheta * cosf(phi);
	H.y = sinTheta * sinf(phi);
	H.z = cosTheta;

	// Create an orthonormal basis (tangent, bitangent, normal)
	float3 up = fabs(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
	float3 tangent = normalize(cross(up, normal));
	float3 bitangent = cross(normal, tangent);

	// Transform the sample direction from local space to world space
	return normalize(tangent * H.x + bitangent * H.y + normal * H.z);
}

__device__ float3 IntegratorPipeline::LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed)
{
	Ray ray = in_ray;

	float3 throughtput = make_float3(1);
	float3 light = make_float3(0);

	ShapeIntersection payload;

	for (int bounce_depth = 0; bounce_depth < globals.IntegratorCFG.bounces; bounce_depth++) {
		seed += bounce_depth;
		payload = Intersect(globals, ray);

		//miss--
		if (payload.hit_distance < 0)
		{
			light += SkyShading(ray) * throughtput;
			break;
		}
		//hit--

		float3 wo = -ray.getDirection();
		light += payload.Le() * throughtput;

		//get BSDF
		BSDF bsdf = payload.getBSDF();

		//sample dir
		float3 wi = sampleCosineWeightedHemisphere(payload.w_norm, get2D_PCGHash(seed));

		float3 fcos = bsdf.f(wo, wi) * AbsDot(wi, payload.w_norm);
		if (!fcos)break;

		float pdf = 1 / (2 * PI);
		pdf = AbsDot(payload.w_norm, wi) / PI;

		throughtput *= fcos / pdf;

		ray = payload.spawnRay(wi);
	}

	return light;
};