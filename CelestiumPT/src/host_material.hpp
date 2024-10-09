#pragma once
#include "glm/glm.hpp"

struct DeviceMaterial;

struct HostMaterial {
	HostMaterial(DeviceMaterial* dmat);
	void updateDevice();

	glm::vec3 albedo_color_factor = glm::vec3(0.8);
	//Texture albedo_color_texture;
	glm::vec3 emission_color_factor = glm::vec3(0.0);
	//Texture emission_color_texture;
	float emission_strength = 1.0f;

	DeviceMaterial* m_device_material = nullptr;
};