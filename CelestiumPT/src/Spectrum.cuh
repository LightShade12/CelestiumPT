#pragma once
#include "maths/vector_maths.cuh"

//Simplified RGB spectral radiance distribution
class RGBSpectrum {
public:
	__device__ __host__ RGBSpectrum() : r(0.f), g(0.f), b(0.f) {};
	__device__ __host__ RGBSpectrum(float r, float g, float b) : r(r), g(g), b(b) {};

	__device__ __host__ explicit RGBSpectrum(float s) : r(s), g(s), b(s) {};

	__device__ __host__ explicit RGBSpectrum(float4 s) : r(s.x), g(s.y), b(s.z) {};
	__device__ __host__ explicit RGBSpectrum(float3 s) : r(s.x), g(s.y), b(s.z) {};

	__device__ __host__ bool operator!() const
	{
		return (r == 0.f && g == 0.f && b == 0.f);
	}

	__device__ __host__ float maxComponentValue() {
		return fmaxf(r, fmaxf(g, b));
	}

	__device__ __host__ operator bool() const
	{
		return (r != 0.f || g != 0.f || b != 0.f);
	}

	__device__ __host__ RGBSpectrum& operator+=(RGBSpectrum s) {
		r += s.r;
		g += s.g;
		b += s.b;
		return *this;
	}

	__device__ __host__ RGBSpectrum operator+(float s) const {
		RGBSpectrum ret = *this;
		return RGBSpectrum(ret.r + s, ret.g + s, ret.b + s);
	}
	__device__ __host__ RGBSpectrum operator-(float s) const {
		RGBSpectrum ret = *this;
		return RGBSpectrum(ret.r - s, ret.g - s, ret.b - s);
	}

	__device__ __host__ RGBSpectrum operator+(RGBSpectrum s) const {
		RGBSpectrum ret = *this;
		return ret += s;
	}

	__device__ __host__ RGBSpectrum& operator-=(RGBSpectrum s) {
		r -= s.r;
		g -= s.g;
		b -= s.b;
		return *this;
	}

	__device__ __host__ RGBSpectrum operator-(RGBSpectrum s) const {
		RGBSpectrum ret = *this;
		return ret -= s;
	}

	__device__ __host__ friend RGBSpectrum operator-(float a, RGBSpectrum s) { return { a - s.r, a - s.g, a - s.b }; }

	__device__ __host__ RGBSpectrum& operator*=(RGBSpectrum s) {
		r *= s.r;
		g *= s.g;
		b *= s.b;
		return *this;
	}

	__device__ __host__ RGBSpectrum operator*(RGBSpectrum s) const {
		RGBSpectrum ret = *this;
		return ret *= s;
	}

	__device__ __host__ RGBSpectrum operator*(float a) const {
		//DCHECK(!IsNaN(a));
		return { a * r, a * g, a * b };
	}

	__device__ __host__ RGBSpectrum& operator*=(float a) {
		//DCHECK(!IsNaN(a));
		r *= a;
		g *= a;
		b *= a;
		return *this;
	}

	__device__ __host__ friend RGBSpectrum operator*(float a, RGBSpectrum s) { return s * a; }

	__device__ __host__ RGBSpectrum& operator/=(RGBSpectrum s) {
		r /= s.r;
		g /= s.g;
		b /= s.b;
		return *this;
	}

	__device__ __host__ RGBSpectrum operator/(RGBSpectrum s) const {
		RGBSpectrum ret = *this;
		return ret /= s;
	}

	__device__ __host__ RGBSpectrum& operator/=(float a) {
		//DCHECK(!IsNaN(a));
		//DCHECK_NE(a, 0);
		r /= a;
		g /= a;
		b /= a;
		return *this;
	}

	__device__ __host__ RGBSpectrum operator/(float a) const {
		RGBSpectrum ret = *this;
		return ret /= a;
	}

	__device__ __host__ RGBSpectrum operator-() const { return { -r, -g, -b }; }

	__device__ __host__ float Average() const { return (r + g + b) / 3; }

	__device__ __host__ bool operator==(RGBSpectrum s) const { return r == s.r && g == s.g && b == s.b; }

	__device__ __host__ bool operator!=(RGBSpectrum s) const { return r != s.r || g != s.g || b != s.b; }

	__device__ __host__ float operator[](int c) const {
		//DCHECK(c >= 0 && c < 3);
		if (c == 0)
			return r;
		else if (c == 1)
			return g;
		return b;
	}

	__device__ __host__ float& operator[](int c) {
		//DCHECK(c >= 0 && c < 3);
		if (c == 0)
			return r;
		else if (c == 1)
			return g;
		return b;
	}

	float r = 0, g = 0, b = 0;
};

//operators
//__device__ __host__ RGBSpectrum operator*(float a, RGBSpectrum b);

//vector types extension
__device__ __host__ float3 make_float3(const RGBSpectrum& rgb);

__device__ __host__ float4 make_float4(const RGBSpectrum& rgb);

__device__ __host__ float4 make_float4(const RGBSpectrum& rgb, float s);

//utility
__device__ RGBSpectrum clampOutput(const RGBSpectrum& rgb);

__device__ float getLuminance(const RGBSpectrum& t_linear_col);