#include "Light.cuh"
//#include "Spectrum.cuh"

struct LightSample {
	LightSample() = default;
	RGBSpectrum L;
	float3 wi{};
	float3 pLight{};
	float pdf = 0;
};

__device__ LightSample Light::SampleLi(LightSampleContext ctx)
{
	return LightSample();
}