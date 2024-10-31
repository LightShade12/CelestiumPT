#include "cuda_utility.cuh"
#include "maths/vector_maths.cuh"
#include "maths/constants.cuh"
#include "storage.cuh"

__device__ float4 texReadNearest(cudaSurfaceObject_t tex_surf, int2 pix) {
	float4 data = surf2Dread<float4>(tex_surf,
		pix.x * (int)sizeof(float4), pix.y);
	return data;
}
//has to be uchar4/2/1 or float4/2/1; no 3 comp color
__device__ void texWrite(float4 data, cudaSurfaceObject_t tex_surf, int2 pix) {
	surf2Dwrite<float4>(data, tex_surf, pix.x * (int)sizeof(float4), pix.y);
}

__device__ void texWrite(float4 data, cudaSurfaceObject_t tex_surf, float2 pix) {
	int2 ipix = make_int2(pix);
	surf2Dwrite<float4>(data, tex_surf, ipix.x * (int)sizeof(float4), ipix.y);
}

__device__ float4 dFdx(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res, int stride) {
	float4 d0 = texReadNearest(data_surfobj, c_pix);
	int2 next = c_pix + make_int2(stride, 0);
	next.x = clamp(next.x, 0, res.x - 1);
	float4 d1 = texReadNearest(data_surfobj, next);
	return d1 - d0;
}

__device__ float4 dFdy(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res, int stride) {
	float4 d0 = texReadNearest(data_surfobj, c_pix);
	int2 next = c_pix + make_int2(0, stride);
	next.y = clamp(next.y, 0, res.y - 1);
	float4 d1 = texReadNearest(data_surfobj, next);
	return d1 - d0;
}

__device__ float4 texReadBilinear(const cudaSurfaceObject_t& tex_surface,
	float2 fpix, int2 t_res, bool lerp_alpha) {
	//TODO:consider half pixel for centre sampling

	int2 pix = make_int2(fpix);//truncate
	int x = pix.x;
	int y = pix.y;

	// Clamp pixel indices to be within bounds
	int s0 = clamp(x, 0, t_res.x - 1);
	int s1 = clamp(x + 1, 0, t_res.x - 1);
	int t0 = clamp(y, 0, t_res.y - 1);
	int t1 = clamp(y + 1, 0, t_res.y - 1);

	//TODO: consider trying unclamped taps for weighting
	// Compute fractional parts for interpolation weights
	float ws = fpix.x - s0;
	float wt = fpix.y - t0;

	// Sample 2x2 texel neighborhood
	float4 cp0 = texReadNearest(tex_surface, { s0, t0 });
	float4 cp1 = texReadNearest(tex_surface, { s1, t0 });
	float4 cp2 = texReadNearest(tex_surface, { s0, t1 });
	float4 cp3 = texReadNearest(tex_surface, { s1, t1 });

	//TODO: replace with lerp
	// Perform bilinear interpolation
	float4 tc0 = cp0 + (cp1 - cp0) * ws;
	float4 tc1 = cp2 + (cp3 - cp2) * ws;
	float4 fc = tc0 + (tc1 - tc0) * wt;

	if (!lerp_alpha) {
		// Nearest neighbor for alpha
		fc.w = (ws > 0.5f ? (wt > 0.5f ? cp3.w : cp1.w) : (wt > 0.5f ? cp2.w : cp0.w));
	}

	return fc;
}

__device__ float texReadGaussianWeighted(cudaSurfaceObject_t t_texture, int2 t_res, int2 t_current_pix) {
	float sum = 0.f;

	const float kernel[2][2] = {
		{ 1.0 / 4.0, 1.0 / 8.0  },
		{ 1.0 / 8.0, 1.0 / 16.0 }
	};

	const int radius = 1;
	for (int yy = -radius; yy <= radius; yy++)
	{
		for (int xx = -radius; xx <= radius; xx++)
		{
			int2 tap_pix = t_current_pix + make_int2(xx, yy);
			tap_pix = clamp(tap_pix, make_int2(0, 0), (t_res - 1));

			float k = kernel[abs(xx)][abs(yy)];

			sum += texReadNearest(t_texture, tap_pix).x * k;
		}
	}

	return sum;
}

__device__ float packStratumPos(int2 pos)
{
	return pos.x + (pos.y * (int)ASVGF_STRATUM_SIZE);
}

__device__ int2 unpackStratumPos(int d)
{
	return make_int2(d % (int)ASVGF_STRATUM_SIZE, d / ASVGF_STRATUM_SIZE);
}