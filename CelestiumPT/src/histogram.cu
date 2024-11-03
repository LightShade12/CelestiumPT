#include "histogram.cuh"

#include "spectrum.cuh"
#include "cuda_utility.cuh"
#include "maths/constants.cuh"
#include "film.cuh"
#include "device_camera.cuh"

#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

// For a given color and luminance range, return the histogram bin index
__device__ uint colorToBin(float3 hdrColor, float minLogLum, float inverseLogLumRange) {
	// Convert our RGB value to Luminance, see note for RGB_TO_LUM macro above
	float lum = getLuminance(RGBSpectrum(hdrColor));

	// Avoid taking the log of zero
	if (lum < HISTOGRAM_LUMINANCE_EPSILON) {
		return 0;
	}

	// Calculate the log_2 luminance and express it as a value in [0.0, 1.0]
	// where 0.0 represents the minimum luminance, and 1.0 represents the max.
	float log_lum = clamp((log2(lum) - minLogLum) * inverseLogLumRange, 0.0, 1.0);

	// Map [0, 1] to [1, 255]. The zeroth bin is handled by the epsilon check above.
	return uint(log_lum * 254.0 + 1.0);
}

//__constant__ constexpr float LOG_LUM_RANGE = 15.f;//TODO: what values for these?
//__constant__ constexpr float MIN_LOG_LUM = -10.f;

__device__ float lumWeight(const IntegratorGlobals& t_globals, int2 pix)
{
	int2 centre_spot = t_globals.FrameBuffer.resolution / 2;

	float distance = length(make_float2(centre_spot - pix));

	float max_distance = fmaxf(t_globals.FrameBuffer.resolution.x, t_globals.FrameBuffer.resolution.y) / 2.0f;//not using width?
	float normalized_distance = distance / max_distance;

	float weight = 1.0f - smoothstep(0.0f, 1.0f, normalized_distance);

	return weight;
}

//Launch with thread dims 16x16=256
__global__ void computeHistogram(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	__shared__ uint shared_histogram[HISTOGRAM_SIZE];

	{
		int2 local_id = make_int2(threadIdx.x, threadIdx.y);
		int local_index = local_id.x + (local_id.y * blockDim.x);
		//----------

		shared_histogram[local_index] = 0;//init
		__syncthreads();

		float3 linear_col = make_float3(texReadNearest(t_globals.FrameBuffer.composite_surfobject,
			current_pix));

		uint bin_idx = colorToBin(linear_col, t_globals.IntegratorCFG.auto_exposure_min_comp,
			(1.f / t_globals.IntegratorCFG.auto_exposure_max_comp));

		atomicAdd(&(shared_histogram[bin_idx]), 1);

		__syncthreads();

		atomicAdd(&(t_globals.GlobalHistogramBuffer[local_index]), shared_histogram[local_index]);
	}
}

//launch with thread dims= 256 x 1;
__global__ void computeAverageLuminance(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	__shared__ uint shared_histogram[HISTOGRAM_SIZE];

	{
		int2 local_id = make_int2(threadIdx.x, threadIdx.y);
		int local_index = local_id.x + (local_id.y * blockDim.x);
		//----------

		// Get the count from the histogram buffer
		uint count_for_this_bin = t_globals.GlobalHistogramBuffer[local_index];
		shared_histogram[local_index] = count_for_this_bin * local_index;//net luminance in the bin

		__syncthreads();

		// Reset the count stored in the buffer in anticipation of the next pass
		t_globals.GlobalHistogramBuffer[local_index] = 0;

		// This loop will perform a weighted count of the luminance range
#pragma unroll
		for (uint cutoff = HISTOGRAM_SIZE / 2; cutoff > 0; cutoff >>= 1) {
			if (uint(local_index) < cutoff) {
				shared_histogram[local_index] += shared_histogram[local_index + cutoff];
			}

			__syncthreads();
		}

		// We only need to calculate this once, so only a single thread is needed.
		if ((current_pix.x + current_pix.y) == 0) {
			// Here we take our weighted sum and divide it by the number of pixels
			// that had luminance greater than zero (since the index == 0, we can
			// use count_for_this_bin to find the number of black pixels)
			float weighted_log_average = (shared_histogram[0] / max((frame_res.x * frame_res.y) - float(count_for_this_bin), 1.0)) - 1.0;

			// Map from our histogram space to actual luminance
			float weighted_avg_lum = exp2(((weighted_log_average / 254.0) * t_globals.IntegratorCFG.auto_exposure_max_comp) +
				t_globals.IntegratorCFG.auto_exposure_min_comp);

			// The new stored value will be interpolated using the last frames value
			// to prevent sudden shifts in the exposure.
			float lum_last_frame = *(t_globals.AverageLuminance);
			float adapted_lum = lum_last_frame + (weighted_avg_lum - lum_last_frame) * t_globals.IntegratorCFG.auto_exposure_speed;//TODO: lerp?
			*(t_globals.AverageLuminance) = adapted_lum;
		}
	}
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

__device__ float3 convertYxy2XYZ(float3 _Yxy)
{
	// Reference(s):
	// - xyY to XYZ
	//   https://web.archive.org/web/20191027010036/http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
	float3 xyz;
	xyz.x = _Yxy.x * _Yxy.y / _Yxy.z;
	xyz.y = _Yxy.x;
	xyz.z = _Yxy.x * (1.0 - _Yxy.y - _Yxy.z) / _Yxy.z;
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

//TODO: standardize color operations
__global__ void toneMap(const IntegratorGlobals t_globals)
{
	//setup threads
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 current_pix = make_int2(thread_pixel_coord_x, thread_pixel_coord_y);

	int2 frame_res = t_globals.FrameBuffer.resolution;
	float2 screen_uv = { (float)current_pix.x / (float)frame_res.x, (float)current_pix.y / (float)frame_res.y };

	if ((current_pix.x >= frame_res.x) || (current_pix.y >= frame_res.y)) return;
	//----------------------------------------------

	float4 col = texReadNearest(t_globals.FrameBuffer.composite_surfobject, current_pix);
	float4 bloom_col = texReadNearest(t_globals.FrameBuffer.bloom_surfobject, current_pix);
	col = lerp(col, bloom_col, t_globals.IntegratorCFG.bloom_lerp);

	RGBSpectrum frag_spectrum = RGBSpectrum(col);

	//TODO: proper calibration
	float exposure = t_globals.SceneDescriptor.ActiveCamera->exposure;
	frag_spectrum *= exposure;

	if (t_globals.IntegratorCFG.auto_exposure_enabled)
	{
		float l_max = (*t_globals.AverageLuminance) * 9.6;
		float3 Yxy = convertRGB2Yxy(make_float3(frag_spectrum));
		Yxy.x = Yxy.x / (l_max);//TODO: add epsilon?
		frag_spectrum = RGBSpectrum(convertYxy2RGB(Yxy));
	}

	//normalize
	//frag_spectrum = toneMapping(frag_spectrum, exposure);
	frag_spectrum = AgxMinimal::agx_fitted(frag_spectrum);

	frag_spectrum = RGBSpectrum(AgxMinimal::agxLook(make_float3(frag_spectrum)));
	//EOTF
	//frag_spectrum = gammaCorrection(frag_spectrum);
	frag_spectrum = AgxMinimal::agx_fitted_Eotf(frag_spectrum);

	float4 frag_color = make_float4(frag_spectrum, 1);

	texWrite(frag_color, t_globals.FrameBuffer.composite_surfobject, current_pix);
}