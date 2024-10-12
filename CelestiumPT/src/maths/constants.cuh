#pragma once
#include <cuda_runtime.h>

__constant__ const constexpr float TRIANGLE_EPSILON = 0.000001f;
__constant__ const constexpr float PI = 3.141592f;
__constant__ const constexpr float HIT_EPSILON = 0.001f;
__constant__ const constexpr float MAT_MIN_ROUGHNESS = 0.045f;