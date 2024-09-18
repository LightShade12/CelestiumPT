#pragma once

//#include "Ray.cuh"
#include "maths/matrix.cuh"
#include <cuda_runtime.h>

class Ray;

class DeviceCamera {
public:

	__device__ __host__ DeviceCamera();

	//screen_uv = -1 to 1
	__device__ Ray generateRay(int frame_width, int frame_height, float2 screen_uv);

	float FOV_y_radians = 1.0472f;
	Mat4 viewMatrix;
	Mat4 projectionMatrix;
	Mat4 invProjectionMatrix;
};