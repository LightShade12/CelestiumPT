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

void IntegratorPipeline::invokeRenderKernel(IntegratorGlobals globals, cudaSurfaceObject_t composite_render_surface_obj, dim3 block_grid_dims, dim3 thread_block_dims, int frame_width, int frame_height)
{
	renderKernel << < block_grid_dims, thread_block_dims >> > (globals, composite_render_surface_obj, frame_width, frame_height);
};

__global__ void renderKernel(IntegratorGlobals globals, cudaSurfaceObject_t composite_render_surface_obj, int frame_width, int frame_height)
{
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((thread_pixel_coord_x >= frame_width) || (thread_pixel_coord_y >= frame_height)) return;

	//float3 outcolor = { (float)thread_pixel_coord_x / frame_width,(float)thread_pixel_coord_y / frame_height,0.5 };
	float3 sampled_radiance = IntegratorPipeline::evaluatePixelSample(globals, { (float)thread_pixel_coord_x,(float)thread_pixel_coord_y },
		frame_width, frame_height);

	float4 fragcolor = { sampled_radiance.x,sampled_radiance.y,sampled_radiance.z, 1 };

	surf2Dwrite(fragcolor, composite_render_surface_obj, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}

__device__ float3 IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel, int frame_width, int frame_height)
{
	uint32_t seed = ppixel.x + ppixel.y * frame_width;
	seed *= globals.frameidx;
	//return make_float3(ppixel.x / frame_width, ppixel.y / frame_height, 0.5);
	return make_float3(get1D_PCGHash(seed), get1D_PCGHash(seed), get1D_PCGHash(seed));
};