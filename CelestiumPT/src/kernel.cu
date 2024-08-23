#include "kernel.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDACC__
#include <surface_indirect_functions.h>

#include <stdio.h>

__global__ void renderKernel(cudaSurfaceObject_t composite_render_surface_obj, int frame_width, int frame_height) {
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((thread_pixel_coord_x >= frame_width) || (thread_pixel_coord_y >= frame_height)) return;

	float3 outcolor = { (float)thread_pixel_coord_x / frame_width,(float)thread_pixel_coord_y / frame_height,0.5 };

	float4 fragcolor = { outcolor.x,outcolor.y,outcolor.z, 1 };

	surf2Dwrite(fragcolor, composite_render_surface_obj, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}

void invokeKernel(cudaSurfaceObject_t composite_render_surface_obj, int frame_width, int frame_height, dim3 block_grid_dims, dim3 thread_block_dims)
{
	renderKernel << < block_grid_dims, thread_block_dims >> > (composite_render_surface_obj, frame_width, frame_height);
}