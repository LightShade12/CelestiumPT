#pragma once
#include "maths/linear_algebra.cuh"
#include <cuda_runtime.h>

class DeviceTexture
{
public:
	__device__ __host__ DeviceTexture(const unsigned char* t_data, int t_width, int t_height, int t_channels);

	__device__ float3 sampleNearest(float2 t_tex_coord, bool t_noncolor) const;
	__device__ float getAlpha(float2 t_texcoords) const;

	__host__ void destroy();

public:
	char m_name[32]{};
private:
	int m_width = 0, m_height = 0, m_channels = 3;
	void* m_data = nullptr;
};