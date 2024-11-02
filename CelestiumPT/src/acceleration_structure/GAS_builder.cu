#include "GAS_builder.hpp"

#include "host_scene.hpp"
#include "device_scene.cuh"
#include "scene_geometry.cuh"
#include "GAS.cuh"

void GASBuilder::build(HostScene* host_scene)
{
	host_scene->syncDeviceGeometry();//sync geometry m_data to device
	GAS gas;
	gas.build(host_scene);
	//TODO: assuming dev_geo is non null
	host_scene->m_DeviceScene->DeviceSceneGeometry->GAS_structure = gas;
	host_scene->syncDeviceGeometry();
}
void GASBuilder::refresh(HostScene* host_scene)
{
	GAS gas = host_scene->m_DeviceScene->DeviceSceneGeometry->GAS_structure;
	gas.refresh(host_scene);
	host_scene->m_DeviceScene->DeviceSceneGeometry->GAS_structure = gas;
	host_scene->syncDeviceGeometry();
};