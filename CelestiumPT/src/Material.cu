#include "HostMaterial.hpp"
#include "DeviceMaterial.cuh"

HostMaterial::HostMaterial(DeviceMaterial* dmat)
{
	m_device_material = dmat;

	albedo_color_factor = glm::vec3(dmat->albedo_color_factor.r, dmat->albedo_color_factor.b, dmat->albedo_color_factor.b);
	emission_color_factor = glm::vec3(dmat->emission_color_factor.r, dmat->emission_color_factor.b, dmat->emission_color_factor.b);
	emission_strength = dmat->emission_strength;
}

void HostMaterial::updateDevice()
{
	if (m_device_material != nullptr) {
		m_device_material->albedo_color_factor =
			RGBSpectrum(albedo_color_factor.r, albedo_color_factor.g, albedo_color_factor.b);
		m_device_material->emission_color_factor =
			RGBSpectrum(emission_color_factor.r, emission_color_factor.g, emission_color_factor.b);
		m_device_material->emission_strength = emission_strength;
	}
}