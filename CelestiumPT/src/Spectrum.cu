#include "Spectrum.cuh"
#include "maths/vector_maths.cuh"

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
}
__device__ RGBSpectrum clampOutput(const RGBSpectrum& rgb)
{
	if ((checkNaN(make_float3(rgb))) || (checkINF(make_float3(rgb))))
		return RGBSpectrum(0);
	else
		return RGBSpectrum(clamp(make_float3(rgb), 0, 1000));
}

//__device__ __host__ RGBSpectrum operator*(float a, RGBSpectrum b)
//{
//	return { a * b.r, a * b.g, a * b.b };
//}