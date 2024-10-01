#pragma once
#include "Spectrum.cuh"

struct DeviceMaterial {
	__device__ DeviceMaterial() = default;
	RGBSpectrum albedo_color_factor = RGBSpectrum(0.8);
	//Texture albedo_color_texture;
	RGBSpectrum emission_color_factor = RGBSpectrum(0.0);
	//Texture emission_color_texture;
	float emission_strength = 1.0f;
};