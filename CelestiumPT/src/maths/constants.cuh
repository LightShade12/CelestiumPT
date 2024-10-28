#pragma once
#include <cuda_runtime.h>

__constant__ constexpr float TRIANGLE_EPSILON = 0.000001f;
__constant__ constexpr float PI = 3.141592f;
__constant__ constexpr float HIT_EPSILON = 0.001f;
__constant__ constexpr float MAT_MIN_ROUGHNESS = 0.045f;
__constant__ constexpr float ASVGF_STRATUM_SIZE = 3;
__constant__ constexpr float HISTOGRAM_LUMINANCE_EPSILON = 0.005;
__constant__ constexpr int HISTOGRAM_SIZE = 256;