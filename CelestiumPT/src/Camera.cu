#include "host_camera.hpp"
#include "device_camera.cuh"
#include "ray.cuh"
#include "maths/linear_algebra.cuh"

HostCamera::HostCamera(DeviceCamera* device_camera)
{
	m_device_camera = device_camera;
	Mat4 invviewmat = m_device_camera->invViewMatrix;
	Mat4 viewmat = m_device_camera->viewMatrix;
	Mat4 projmat = m_device_camera->projectionMatrix;
	Mat4 invprojmat = m_device_camera->invProjectionMatrix;

	m_view = viewmat.toGLM();
	m_transform = invviewmat.toGLM();
	m_projection = projmat.toGLM();
	m_invProjection = invprojmat.toGLM();

	fov_y_radians = device_camera->fov_y_radians;
}

void HostCamera::updateDevice()
{
	if (m_device_camera != nullptr) {
		Mat4 invviewmat(m_transform);
		Mat4 viewmat(m_view);
		Mat4 projmat(m_projection);
		Mat4 invprojmat(m_invProjection);

		//m_device_camera->prev_projectionMatrix = m_device_camera->projectionMatrix;
		//m_device_camera->prev_viewMatrix = m_device_camera->viewMatrix;

		m_device_camera->exposure = exposure;
		m_device_camera->invViewMatrix = invviewmat;
		m_device_camera->viewMatrix = viewmat;
		m_device_camera->projectionMatrix = projmat;
		m_device_camera->invProjectionMatrix = invprojmat;

		m_device_camera->fov_y_radians = fov_y_radians;
	}
}

void HostCamera::updateCamera()
{
	if (m_device_camera != nullptr) {
		m_device_camera->prev_projectionMatrix = m_device_camera->projectionMatrix;
		m_device_camera->prev_viewMatrix = m_device_camera->viewMatrix;
	}
}

__device__ DeviceCamera::DeviceCamera()
{
	invViewMatrix = Mat4(
		make_float4(1, 0, 0, 0),
		make_float4(0, 1, 0, 0),
		make_float4(0, 0, -1, 0),
		make_float4(0, 0, 0, 0)
	);
	viewMatrix = Mat4(1);
	projectionMatrix = Mat4(1);
	prev_viewMatrix = viewMatrix;
	prev_projectionMatrix = projectionMatrix;

	fov_y_radians = deg2rad(60);
};

__device__ Ray DeviceCamera::generateRay(int frame_width, int frame_height, float2 screen_uv) const
{
	float4 target_cs = invProjectionMatrix * make_float4(screen_uv.x, screen_uv.y, 1.f, 1.f);
	float4 target_ws = invViewMatrix * make_float4(
		normalize(make_float3(target_cs) / target_cs.w), 0);
	float3 raydir_ws = make_float3(target_ws);
	float3 rayorig_ws = make_float3(invViewMatrix * make_float4(0, 0, 0, 1));

	return Ray(rayorig_ws, raydir_ws);
};