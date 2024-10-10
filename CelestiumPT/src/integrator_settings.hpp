#pragma once

struct IntegratorSettings {
	bool accumulate = false;
	bool temporal_accumulation = false;
	bool use_SVGF = false;
	bool use_5x5_filter = false;

	int max_bounces = 2;
	bool MIS = false;
	float GAS_shading_brightness = 0.1;
};