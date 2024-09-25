#pragma once

struct IntegratorSettings {
	bool accumulate = true;
	bool temporal_accumulation = false;
	int bounces = 2;
	bool MIS = false;
	float GAS_shading_brightness = 0.1;
};