#include "denoiser.cuh"
#include "cuda_utility.cuh"
#include "maths/constants.cuh"
#include "maths/linear_algebra.cuh"

__global__ void createGradientSamples(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x / ASVGF_STRATUM_SIZE) || (current_pix.y >= frame_res.y / ASVGF_STRATUM_SIZE)) return;
	//=========================================================

	int2 grad_pix = current_pix;

	float4 gradFB = texReadNearest(t_globals.FrameBuffer.asvgf_gradient_sample_surfobject, grad_pix);

	int2 pos_in_stratum = unpackStratumPos(gradFB.x);
	int2 sampling_pix = grad_pix * ASVGF_STRATUM_SIZE + pos_in_stratum;
	sampling_pix = clamp(sampling_pix, { 0,0 }, frame_res - 1);

	int current_objID = texReadNearest(t_globals.FrameBuffer.objectID_surfobject, sampling_pix).x;
	float4 seed = texReadNearest(t_globals.FrameBuffer.seeds_surfobject, sampling_pix);

	//void sample/ miss/ sky; invalid grad sample
	if (current_objID < 0 || seed.w < 0)
	{
		texWrite(make_float4(0, 0, 0, 0),
			t_globals.FrameBuffer.asvgf_sparse_gradient_surfobject, grad_pix);
		return;
	}

	//init luminance estimates

	float stratum_luminance_estimate = 0;
	float stratum_variance_estimate = 0;

	float s = 0, s2 = 0;
	constexpr int stratum_radius = ASVGF_STRATUM_SIZE / 2.f;
	int samples = 0;

	//stratum scan
	for (int y = -stratum_radius; y <= stratum_radius; y++) {
		for (int x = -stratum_radius; x <= stratum_radius; x++)
		{
			int2 offset = make_int2(x, y);
			int2 tap_pix = sampling_pix + offset;
			tap_pix = clamp(tap_pix, make_int2(0, 0), (frame_res - 1));

			//at this point history contains current frame shading
			float3 color = make_float3(texReadNearest(t_globals.FrameBuffer.raw_irradiance_surfobject, tap_pix));
			float lum = getLuminance(RGBSpectrum(color));

			//TODO: include weight factor for objectID check
			stratum_luminance_estimate += lum;
			s += lum;
			s2 += Sqr(lum);

			samples++;
		}
	}

	stratum_luminance_estimate /= samples;
	stratum_variance_estimate = fabsf((s2 / samples) - Sqr(s / samples));

	//full-res samples
	float4 current_shading = texReadNearest(t_globals.FrameBuffer.raw_irradiance_surfobject, sampling_pix);
	float4 prev_shading = texReadNearest(t_globals.FrameBuffer.history_shading_surfobject, sampling_pix);

	float c_lum = getLuminance(RGBSpectrum(current_shading));
	float p_lum = getLuminance(RGBSpectrum(prev_shading));

	float delta = c_lum - p_lum;
	float normalization_factor = fmaxf(c_lum, p_lum);

	//for debugging
	float3 delta_col = make_float3((delta < 0) ? fabsf(delta) : 0,
		(delta > 0) ? fabsf(delta) : 0,
		0);
	//delta_col = make_float3(fabsf(delta));

	texWrite(make_float4(delta_col, normalization_factor),
		t_globals.FrameBuffer.asvgf_sparse_gradient_surfobject, grad_pix);
	texWrite(make_float4(make_float3(stratum_luminance_estimate), 1),
		t_globals.FrameBuffer.asvgf_luminance_front_surfobject, grad_pix);
	texWrite(make_float4(make_float3(stratum_variance_estimate), 1),
		t_globals.FrameBuffer.asvgf_variance_front_surfobject, grad_pix);
}