#include "SVGFEdgeStoppingFunctions.cuh"
#include "maths/vector_maths.cuh"

// Normal-weighting function (4.4.1)
__device__ float normalWeight(float3 normal0, float3 normal1) {
	const float exponent = 64.0;
	return pow(max(0.0, dot(normal0, normal1)), exponent);
}

// Depth-weighting function (4.4.2)
__device__ float depthWeight(float depth0, float depth1, float2 grad, float2 offset) {
	// paper uses eps = 0.005 for a normalized depth buffer
	// ours is not but 0.1 seems to work fine
	const float eps = 0.1;
	const float SIGMA_Z = 1;
	return exp((-abs(depth0 - depth1)) / (SIGMA_Z * (abs(dot(grad, offset)) + eps)));
}

// Luminance-weighting function (4.4.3)
__device__ float luminanceWeight(float lum0, float lum1, float variance) {
	const float strictness = 4.0;//SIGMA_L
	const float eps = 0.01;
	return exp((-abs(lum0 - lum1)) / (strictness * sqrtf(variance) + eps));
}