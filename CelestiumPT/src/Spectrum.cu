#include "spectrum.cuh"
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

__device__ float getLuminance(const RGBSpectrum& col) {
	// Rec. 709 luminance coefficients for linear RGB
	return 0.2126f * col.r + 0.7152f * col.g + 0.0722f * col.b;
}
__device__ float3 convertRGB2XYZ(float3 _rgb)
{
	// Reference(s):
	// - RGB/XYZ Matrices
	//   https://web.archive.org/web/20191027010220/http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
	float3 xyz;
	xyz.x = dot(make_float3(0.4124564, 0.3575761, 0.1804375), _rgb);
	xyz.y = dot(make_float3(0.2126729, 0.7151522, 0.0721750), _rgb);
	xyz.z = dot(make_float3(0.0193339, 0.1191920, 0.9503041), _rgb);
	return xyz;
}

__device__ float3 convertXYZ2RGB(float3 _xyz)
{
	float3 rgb;
	rgb.x = dot(make_float3(3.2404542, -1.5371385, -0.4985314), _xyz);
	rgb.y = dot(make_float3(-0.9692660, 1.8760108, 0.0415560), _xyz);
	rgb.z = dot(make_float3(0.0556434, -0.2040259, 1.0572252), _xyz);
	return rgb;
}

__device__ float3 convertXYZ2Yxy(float3 _xyz)
{
	// Reference(s):
	// - XYZ to xyY
	//   https://web.archive.org/web/20191027010144/http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
	float inv = 1.0 / dot(_xyz, make_float3(1.0, 1.0, 1.0));
	return make_float3(_xyz.y, _xyz.x * inv, _xyz.y * inv);
}

__device__ float3 convertYxy2XYZ(float3 Yxy_)
{
	// Reference(s):
	// - xyY to XYZ
	//   https://web.archive.org/web/20191027010036/http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
	float3 xyz;
	xyz.x = Yxy_.x * Yxy_.y / Yxy_.z;
	xyz.y = Yxy_.x;
	xyz.z = Yxy_.x * (1.0 - Yxy_.y - Yxy_.z) / Yxy_.z;
	return xyz;
}

__device__ float3 convertRGB2Yxy(float3 _rgb)
{
	return convertXYZ2Yxy(convertRGB2XYZ(_rgb));
}

__device__ float3 convertYxy2RGB(float3 _Yxy)
{
	return convertXYZ2RGB(convertYxy2XYZ(_Yxy));
}