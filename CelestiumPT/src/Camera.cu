#include "HostCamera.hpp"
#include "DeviceCamera.cuh"
#include "Ray.cuh"
#include "maths/maths_linear_algebra.cuh"

HostCamera::HostCamera(DeviceCamera* device_camera)
{
	m_device_camera = device_camera;
	Mat4 viewmat = m_device_camera->viewMatrix;
	Mat4 projmat = m_device_camera->projectionMatrix;
	Mat4 invprojmat = m_device_camera->invProjectionMatrix;

	m_transform = viewmat.toGLM();
	m_projection = projmat.toGLM();
	m_invProjection = invprojmat.toGLM();

	FOV_y_radians = device_camera->FOV_y_radians;
}

void HostCamera::updateDevice()
{
	if (m_device_camera != nullptr) {
		Mat4 viewmat(m_transform);
		Mat4 projmat(m_projection);
		Mat4 invprojmat(m_invProjection);

		m_device_camera->viewMatrix = viewmat;
		m_device_camera->projectionMatrix = projmat;
		m_device_camera->invProjectionMatrix = invprojmat;

		m_device_camera->FOV_y_radians = FOV_y_radians;
	}
}

__device__ DeviceCamera::DeviceCamera()
{
	viewMatrix = Mat4(
		make_float4(1, 0, 0, 0),
		make_float4(0, 1, 0, 0),
		make_float4(0, 0, -1, 0),
		make_float4(0, 0, 0, 0)
	);
	projectionMatrix = Mat4(1);
	FOV_y_radians = deg2rad(60);
};

__device__ Ray DeviceCamera::generateRay(int frame_width, int frame_height, float2 screen_uv)
{
	float theta = FOV_y_radians / 2.f;

	float fov_factor = tanf(theta);

	float aspect_ratio = float(frame_width) / float(frame_height);

	float4 target_cs = invProjectionMatrix * make_float4(screen_uv.x, screen_uv.y, 1.f, 1.f);
	float4 target_ws = viewMatrix * make_float4(
		normalize(make_float3(target_cs) / target_cs.w), 0);
	float3 raydir_ws = (make_float3(target_ws));
	float3 rayorig_ws = make_float3(viewMatrix * make_float4(0, 0, 0, 1));

	return Ray(rayorig_ws, raydir_ws);
};