#pragma once

struct IntegratorSettings {
	bool accumulate = false;
	bool temporal_filter_enabled = false;
	bool adaptive_temporal_filter_enabled = false;

	bool svgf_enabled = true;
	bool use_5x5_filter = false;

	bool sunlight_enabled = false;
	float sunlight_intensity = 6;
	float sun_theta = 0.785;
	float sun_phi = -0.890118;//rads
	//float sun_distance = 90;//TODO:remove safely
	float sun_angular_diameter_radians = 0.00872665;//0.5 deg

	bool skylight_enabled = true;
	float skylight_intensity = 790;
	float rl_coeff_r = 1;
	float rl_coeff_g = 1;
	float rl_coeff_b = 1;

	float bloom_lerp = 0.3;
	float bloom_internal_lerp = 0.75;

	bool auto_exposure_enabled = true;
	float auto_exposure_speed = 0.1f;
	float auto_exposure_max_comp = 20.f;//-12.47393f
	float auto_exposure_min_comp = -5.5f;//4.026069f;

	int max_bounces = 2;
	bool MIS = false;

	float GAS_shading_brightness = 0.1;
};