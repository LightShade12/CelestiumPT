#include "GAS.cuh"
#include "HostScene.hpp"
#include "DeviceScene.cuh"
#include "Storage.cuh"
#include "ShapeIntersection.cuh"

void GAS::build(HostScene* host_scene)
{
	blasbuilder.build(host_scene);
	DeviceScene* dscene = host_scene->m_DeviceScene;
	std::vector<TLASNode>tlasnodes;
	tlas = TLAS(dscene->DeviceBLASes, tlasnodes);
	dscene->DeviceTLASNodes = tlasnodes;
	host_scene->syncDeviceGeometry();
}
__device__ ShapeIntersection GAS::intersect(const IntegratorGlobals& globals, const Ray& ray)
{
	ShapeIntersection payload;
	payload.hit_distance = FLT_MAX;
	tlas.intersect(globals, ray, &payload);
	return payload;
}
;