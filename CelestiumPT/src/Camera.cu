#include "HostCamera.hpp"
#include "DeviceCamera.cuh"
#include "Ray.cuh"

HostCamera::HostCamera(DeviceCamera* dev_camera)
{
	m_device_camera = dev_camera;
	Mat4 mat = m_device_camera->viewMatrix;
	m_transform = glm::mat4(
		mat[0].x, mat[0].y, mat[0].z, mat[0].w,
		mat[1].x, mat[1].y, mat[1].z, mat[1].w,
		mat[2].x, mat[2].y, mat[2].z, mat[2].w,
		mat[3].x, mat[3].y, mat[3].z, mat[3].w
	);
}

void HostCamera::updateDevice()
{
	if (m_device_camera != nullptr) {
		Mat4 mat(
			m_transform[0][0], m_transform[0][1], m_transform[0][2], m_transform[0][3],  // First column
			m_transform[1][0], m_transform[1][1], m_transform[1][2], m_transform[1][3],  // Second column
			m_transform[2][0], m_transform[2][1], m_transform[2][2], m_transform[2][3],  // Third column
			m_transform[3][0], m_transform[3][1], m_transform[3][2], m_transform[3][3]   // Fourth column
		);
		m_device_camera->viewMatrix = mat;
	}
}

__host__ __device__ float deg2rad(float degree)
{
	float const PI = 3.14159265359f;
	return (degree * (PI / 180.f));
}

__device__ Ray DeviceCamera::generateRay(int frame_width, int frame_height, float2 screen_uv)
{
	float3 position = make_float3(viewMatrix[3]);
	float3 forward = make_float3(viewMatrix[2]);
	float3 up = make_float3(viewMatrix[1]);
	float3 right = make_float3(viewMatrix[0]);

	float vertical_fov_radians = deg2rad(60);

	float theta = vertical_fov_radians / 2;

	float fov_factor = tanf(theta / 2.0f);

	float aspect_ratio = (float)frame_width / frame_height;

	float film_plane_height = 2.0f * fov_factor;
	float film_plane_width = film_plane_height * aspect_ratio;
	float film_plane_distance = 1;

	//doing it like this intead of matmul circumvents the issue of translated raydir
	float3 sample_pt =
		(right * film_plane_width * screen_uv.x) +
		(up * film_plane_height * screen_uv.y) +
		(forward * film_plane_distance);

	float3 raydir = normalize(sample_pt);

	return Ray(position, raydir);
};