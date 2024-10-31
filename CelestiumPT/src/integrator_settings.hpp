#pragma once

struct IntegratorSettings {
	bool accumulate = false;
	bool temporal_filter_enabled = false;
	bool adaptive_temporal_filter_enabled = false;
	bool svgf_enabled = false;
	bool use_5x5_filter = false;
	bool auto_exposure_enabled = true;

	bool sunlight_enabled = false;
	float sunlight_intensity = 20;

	bool skylight_enabled = true;
	float skylight_intensity = 1;

	float auto_exposure_speed = 0.1f;
	float auto_exposure_max_comp = 20.f;
	float auto_exposure_min_comp = -5.5f;
	int max_bounces = 2;
	bool MIS = false;
	float GAS_shading_brightness = 0.1;
};