#pragma once
#include "Ray.cuh"

class DeviceCamera {
public:

	__device__ DeviceCamera() {
		viewMatrix = Mat4(
			make_float4(1, 0, 0, 0),
			make_float4(0, 1, 0, 0),
			make_float4(0, 0, -1, 0),
			make_float4(0, 0, 0, 0)
		);
	};

	//screen_uv = -1 to 1
	__device__ Ray generateRay(int frame_width, int frame_height, float2 screen_uv);

	float3 position{};
	float3 forward{};
	float3 up{};
	float3 right{};

	Mat4 viewMatrix;
};