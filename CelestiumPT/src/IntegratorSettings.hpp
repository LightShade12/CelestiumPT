#pragma once

struct IntegratorSettings {
	bool accumulate = true;
	int bounces = 2;
	bool MIS = false;
	float GAS_shading_brightness = 0.1;
};