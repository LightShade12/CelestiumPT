#pragma once
#include "spectrum.cuh"

struct DeviceMaterial {
	__device__ DeviceMaterial() = default;
	RGBSpectrum albedo_color_factor = RGBSpectrum(0.8);
	int albedo_color_texture_id = -1;
	RGBSpectrum emission_color_factor = RGBSpectrum(0.0);
	//Texture emission_color_texture;
	float emission_strength = 1.0f;
};