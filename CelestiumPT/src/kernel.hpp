#pragma once
#include <glad/glad.h>
#include <cuda_gl_interop.h>

void invokeKernel(cudaSurfaceObject_t composite_render_surface_obj, int frame_width,
	int frame_height, dim3 block_grid_dims, dim3 thread_block_dims);