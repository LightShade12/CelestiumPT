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

__constant__ constexpr float LOG_LUM_RANGE = 15.f;//TODO: what values for these?
__constant__ constexpr float MIN_LOG_LUM = -10.f;
__constant__ constexpr float TIME_COEFF = 0.1;

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

		uint bin_idx = colorToBin(linear_col, MIN_LOG_LUM, 1.f / LOG_LUM_RANGE);

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
			float weightedLogAverage = (shared_histogram[0] / max((frame_res.x * frame_res.y) - float(count_for_this_bin), 1.0)) - 1.0;

			// Map from our histogram space to actual luminance
			float weightedAvgLum = exp2(((weightedLogAverage / 254.0) * LOG_LUM_RANGE) + MIN_LOG_LUM);

			// The new stored value will be interpolated using the last frames value
			// to prevent sudden shifts in the exposure.
			float lumLastFrame = *(t_globals.AverageLuminance);
			float adaptedLum = lumLastFrame + (weightedAvgLum - lumLastFrame) * TIME_COEFF;//TODO: lerp?
			*(t_globals.AverageLuminance) = adaptedLum;
		}
	}
}

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

	RGBSpectrum frag_spectrum = RGBSpectrum(col);

	float exposure = t_globals.SceneDescriptor.ActiveCamera->exposure;
	exposure /= *(t_globals.AverageLuminance);

	//normalize
	frag_spectrum = toneMapping(frag_spectrum, exposure);
	//EOTF
	frag_spectrum = gammaCorrection(frag_spectrum);

	float4 frag_color = make_float4(frag_spectrum, 1);

	texWrite(frag_color, t_globals.FrameBuffer.composite_surfobject, current_pix);
}