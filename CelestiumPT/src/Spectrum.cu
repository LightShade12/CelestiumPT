#include "Spectrum.cuh"

//vector types extension------------

__device__ __host__ float3 make_float3(const RGBSpectrum& rgb)
{
	return make_float3(rgb.r, rgb.g, rgb.b);
};
__device__ __host__ float4 make_float4(const RGBSpectrum& rgb)
{
	return make_float4(rgb.r, rgb.g, rgb.b, 0.f);
}
__device__ __host__ float4 make_float4(const RGBSpectrum& rgb, float s)
{
	return make_float4(rgb.r, rgb.g, rgb.b, s);
};