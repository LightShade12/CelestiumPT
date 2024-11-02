#include "device_texture.cuh"

__device__ __host__ DeviceTexture::DeviceTexture(const unsigned char* t_data, int t_width, int t_height, int t_channels) :
	m_channels(t_channels), m_width(t_width), m_height(t_height)
{
	size_t bytes = t_width * t_height * t_channels * sizeof(unsigned char);
	cudaMalloc(&m_data, bytes);
	cudaMemcpy(m_data, t_data, bytes, cudaMemcpyHostToDevice);
};

__device__ float3 DeviceTexture::sampleNearest(float2 t_tex_coord, bool t_noncolor) const
{
	// Wrap UV coordinates to [0, 1) range
	t_tex_coord.x = t_tex_coord.x - floorf(t_tex_coord.x);
	t_tex_coord.y = t_tex_coord.y - floorf(t_tex_coord.y);

	int x = (t_tex_coord.x * m_width);
	int y = (t_tex_coord.y * m_height);

	int index = x + y * m_width;

	uchar4 fcol = { 0,0,0,255 };

	if (m_channels == 3) {
		uchar3* coldata = (uchar3*)m_data;
		uchar3 rgb = (coldata[y * m_width + x]);//bad var name
		fcol = { rgb.x,rgb.y,rgb.z , 255 };
	}
	else {
		uchar4* coldata = (uchar4*)m_data;
		fcol = (coldata[y * m_width + x]);
	}

	float3 fltcol = make_float3(fcol.x / (float)255, fcol.y / (float)255, fcol.z / (float)255);

	if (!t_noncolor)
		fltcol = { powf(fltcol.x,2), powf(fltcol.y,2), powf(fltcol.z,2) };//srgb to linear
	return fltcol;
}

__device__ float DeviceTexture::getAlpha(float2 t_texcoords) const
{
	if (m_channels < 4)
		return 1;

	int x = (t_texcoords.x - floorf(t_texcoords.x)) * m_width;//wrapping
	int y = (t_texcoords.y - floorf(t_texcoords.y)) * m_height;

	uchar4* coldata = (uchar4*)m_data;
	unsigned char fcol = (coldata[y * m_width + x]).w;

	float alpha = fcol / (float)255;

	return alpha;//normalized
}

__host__ void DeviceTexture::destroy()
{
	printf("\ntex destroyed\n");
	if (m_data != nullptr) {
		cudaFree(m_data);
		m_data = nullptr;
	}
}