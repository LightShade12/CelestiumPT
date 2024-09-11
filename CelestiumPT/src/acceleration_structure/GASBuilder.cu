#include "GASBuilder.hpp"

#include "HostScene.hpp"
#include "DeviceScene.cuh"
#include "SceneGeometry.cuh"
#include "GAS.cuh"

void GASBuilder::build(HostScene* host_scene)
{
	GAS gas;
	gas.build(host_scene);
	//TODO: assuming dev_geo is non null
	host_scene->m_DeviceScene->DeviceSceneGeometry->GAS_structure = gas;
	//TODO: automatically call sync?
}
void GASBuilder::refresh(HostScene* host_scene)
{
	GAS gas = host_scene->m_DeviceScene->DeviceSceneGeometry->GAS_structure;
	gas.refresh(host_scene);
	host_scene->m_DeviceScene->DeviceSceneGeometry->GAS_structure = gas;
	host_scene->syncDeviceGeometry();
};