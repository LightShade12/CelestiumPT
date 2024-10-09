#define __CUDACC__
#include <cuda_runtime.h>
#include <vector_types.h>

struct IntegratorGlobals;
// Functions for texture read/write operations
__device__ float4 texReadNearest(cudaSurfaceObject_t tex_surf, int2 pix);
__device__ void texWrite(float4 data, cudaSurfaceObject_t tex_surf, int2 pix);
__device__ void texWrite(float4 data, cudaSurfaceObject_t tex_surf, float2 pix);

// Functions to compute screen-space derivatives
__device__ float4 dFdx(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res);
__device__ float4 dFdy(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res);

__device__ float4 texReadBilinear(const cudaSurfaceObject_t& tex_surface,
	float2 fpix, int2 t_res, bool lerp_alpha);

// computes a 3x3 gaussian blur of the texture_data, centered around
// the current pixel
__device__ float texReadGaussianWeighted(cudaSurfaceObject_t t_texture, int2 t_res, int2 t_current_pix);