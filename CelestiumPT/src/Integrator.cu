#include "Integrator.cuh"
#include "device_launch_parameters.h"
#define __CUDACC__
#include <surface_indirect_functions.h>

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

struct Hitpayload {
	float dist = -1;
	float3 w_pos{};
	float3 w_norm{};
};

__device__ Hitpayload hitsphere(const Ray& ray) {
	float3 center = { 0,0,-20 };
	float radius = 5;
	float3 oc = center - ray.getOrigin();
	float a = dot(ray.getDirection(), ray.getDirection());
	float b = -2.0 * dot(ray.getDirection(), oc);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4 * a * c;

	Hitpayload payload;
	if (discriminant < 0) {
		payload.dist = -1;
	}
	else {
		payload.dist = (-b - sqrtf(discriminant)) / (2.0 * a);
		payload.w_pos = ray.getOrigin() + (ray.getDirection() * payload.dist);
		payload.w_norm = normalize(payload.w_pos - center);
	}
	return payload;
}

__device__ float3 IntegratorPipeline::Li(const IntegratorGlobals& globals, const Ray& ray)
{
	Hitpayload hitpayload = hitsphere(ray);

	if (hitpayload.dist < 0) {
		//miss
		float3 unit_direction = normalize(ray.getDirection());
		float a = 0.5f * (unit_direction.y + 1.0);
		return (1.0f - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
	}
	//hit
	return hitpayload.w_norm;

	return make_float3(1, 0, 0);
};