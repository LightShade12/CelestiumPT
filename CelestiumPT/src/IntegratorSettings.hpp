#pragma once

struct IntegratorSettings {
	bool accumulate = false;
	bool temporal_accumulation = true;

	int max_bounces = 2;
	bool MIS = false;
	float GAS_shading_brightness = 0.1;
};