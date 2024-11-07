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

__device__ float meteringWeight(const IntegratorGlobals& t_globals, int2 pix)
{
	//return 1.0f;

	int2 centre_spot = t_globals.FrameBuffer.resolution / 2;

	constexpr float radius_factor = 1.0f;//TODO: can be user param

	float distance = length(make_float2(centre_spot - pix));
	distance *= (1 / radius_factor);

	float max_distance = fminf(t_globals.FrameBuffer.resolution.x, t_globals.FrameBuffer.resolution.y) / 2.0f;//not using width?
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

	__shared__ float shared_histogram[HISTOGRAM_SIZE];

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

		float w = meteringWeight(t_globals, current_pix);

		w = bin_idx > 0 ? w : 1;//store pixels count for black pixels instead of weights sum

		texWrite(make_float4(make_float3(w), 1), t_globals.FrameBuffer.debugview_AECW_surfobject, current_pix);

		atomicAdd(&(shared_histogram[bin_idx]), w);

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

	__shared__ float shared_weighted_luminance_histogram[HISTOGRAM_SIZE];
	__shared__ float shared_net_weights[HISTOGRAM_SIZE];

	{
		int2 local_id = make_int2(threadIdx.x, threadIdx.y);
		int local_index = local_id.x + (local_id.y * blockDim.x);
		//----------

		// Get the count from the histogram buffer
		float weights_sum_for_this_bin = t_globals.GlobalHistogramBuffer[local_index];
		const int& bin_luminance_value = local_index;

		shared_weighted_luminance_histogram[local_index] = weights_sum_for_this_bin * bin_luminance_value;//net metering weighted luminance in the bin
		shared_net_weights[local_index] = weights_sum_for_this_bin;

		__syncthreads();

		// Reset the count stored in the buffer in anticipation of the next pass
		t_globals.GlobalHistogramBuffer[local_index] = 0;

		// This loop will perform a weighted count of the luminance range
#pragma unroll
		for (uint cutoff = HISTOGRAM_SIZE / 2; cutoff > 0; cutoff >>= 1) {
			if (uint(local_index) < cutoff) {
				shared_weighted_luminance_histogram[local_index] += shared_weighted_luminance_histogram[local_index + cutoff];
				shared_net_weights[local_index] += shared_net_weights[local_index + cutoff];
			}

			__syncthreads();
		}

		// We only need to calculate this once, so only a single thread is needed.
		if (local_index == 0) {
			// Here we take our weighted sum and divide it by the number of pixels
			// that had luminance greater than zero (since the index == 0, we can
			// use count_for_this_bin to find the number of black pixels)
			//float denom = max((frame_res.x * frame_res.y) - float(weights_sum_for_this_bin), 1.0);//works cuz we dont store weight sum for black px
			float denom = max(shared_net_weights[0] - float(weights_sum_for_this_bin), 1.0);//works cuz we dont store weight sum for black px
			float weighted_log_average = (shared_weighted_luminance_histogram[0] / denom) - 1.0;

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

//TODO:proper exposure-luminance calculations
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
	//float exposure_comp_EV = t_globals.SceneDescriptor.ActiveCamera->exposure;
	//frag_spectrum *= powf(2, exposure_comp_EV);

	frag_spectrum *= exposure;

	constexpr float LUMINANCE_EPSILON = 0.1;

	if (t_globals.IntegratorCFG.auto_exposure_enabled)
	{
		//ISO=100
		//K=12.5

		float l_max = ((*t_globals.AverageLuminance) * 9.6) + LUMINANCE_EPSILON;
		float H = 1 / l_max;
		float3 Yxy = convertRGB2Yxy(make_float3(frag_spectrum));
		Yxy.x = Yxy.x * H;//scale luminance

		frag_spectrum = RGBSpectrum(convertYxy2RGB(Yxy));
	}
	//DEBUG
	//if (current_pix == frame_res / 2) {
	//	printf("color: %.3f %.3f %.3f\n", frag_spectrum.r, frag_spectrum.g, frag_spectrum.b);
	//}

	//normalize
	//frag_spectrum = toneMapping(frag_spectrum, 1);
	frag_spectrum = clampOutput(frag_spectrum);

	frag_spectrum = AgxMinimal::agx_fitted(frag_spectrum);

	frag_spectrum = RGBSpectrum(AgxMinimal::agxLook(make_float3(frag_spectrum)));

	//EOTF
	//frag_spectrum = gammaCorrection(frag_spectrum);
	frag_spectrum = AgxMinimal::agx_fitted_Eotf(frag_spectrum);

	float4 frag_color = make_float4(frag_spectrum, 1);

	texWrite(frag_color, t_globals.FrameBuffer.composite_surfobject, current_pix);
}