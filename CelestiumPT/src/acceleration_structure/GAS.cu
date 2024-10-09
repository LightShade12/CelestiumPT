#include "GAS.cuh"
#include "host_scene.hpp"
#include "device_scene.cuh"
#include "storage.cuh"
#include "shape_intersection.cuh"

void GAS::build(HostScene* host_scene)
{
	blasbuilder.build(host_scene);
	DeviceScene* dscene = host_scene->m_DeviceScene;
	std::vector<TLASNode>tlasnodes;
	tlas = TLAS(dscene->DeviceBLASes, tlasnodes);
	dscene->DeviceTLASNodes = tlasnodes;
	host_scene->syncDeviceGeometry();
}

void GAS::refresh(HostScene* host_scene)
{
	DeviceScene* dscene = host_scene->m_DeviceScene;
	std::vector<TLASNode>tlasnodes;//for reconstruction
	tlas.refresh(dscene->DeviceBLASes, tlasnodes);
	dscene->DeviceTLASNodes = tlasnodes;
	host_scene->syncDeviceGeometry();
}
__device__ ShapeIntersection GAS::intersect(const IntegratorGlobals& globals, const Ray& ray, float tmax)
{
	ShapeIntersection payload;
	payload.hit_distance = tmax;
	tlas.intersect(globals, ray, &payload);
	return payload;
}
__device__ bool GAS::intersectP(const IntegratorGlobals& globals, const Ray& ray, float tmax)
{
	return tlas.intersectP(globals, ray, tmax);
}
;