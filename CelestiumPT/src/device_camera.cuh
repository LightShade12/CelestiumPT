#pragma once

#include "maths/matrix_maths.cuh"
#include <cuda_runtime.h>

class Ray;

class DeviceCamera {
public:

	__device__ __host__ DeviceCamera();

	//screen_uv = -1 to 1
	__device__ Ray generateRay(int frame_width, int frame_height, float2 screen_uv) const;

	float fov_y_radians = 1.0472f;
	float exposure = 1;
	Mat4 viewMatrix, prev_viewMatrix;
	Mat4 invViewMatrix;
	Mat4 projectionMatrix, prev_projectionMatrix;
	Mat4 invProjectionMatrix;
};