#pragma once

#define __CUDACC__
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>

// Forward declarations of structures
struct IntegratorGlobals;
struct GBuffer;
class RGBSpectrum;

// Function to compute luminance from RGBSpectrum
__device__ float getLuminance(const RGBSpectrum& col);

// Functions for texture read/write operations
__device__ float4 texRead(cudaSurfaceObject_t tex_surf, int2 pix);
__device__ void texWrite(float4 data, cudaSurfaceObject_t tex_surf, int2 pix);

// Functions to compute screen-space derivatives
__device__ float4 dFdx(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res);
__device__ float4 dFdy(cudaSurfaceObject_t data_surfobj, int2 c_pix, int2 res);

// Path tracing kernel that writes irradiance, moment data, and G-buffer
__global__ void renderPathTraceRaw(IntegratorGlobals globals);

// Temporal integration kernel for SVGF
__global__ void temporalIntegrate(IntegratorGlobals globals);

// SVGF kernel that applies spatial filtering
__global__ void SVGFPass(const IntegratorGlobals globals, int stepsize);

// Function to compose the final composite image
__global__ void composeCompositeImage(const IntegratorGlobals globals);

// Normal-weighting function (Section 4.4.1)
__device__ float normalWeight(float3 normal0, float3 normal1);

// Depth-weighting function (Section 4.4.2)
__device__ float depthWeight(float depth0, float depth1, float2 grad, float2 offset);

// Luminance-weighting function (Section 4.4.3)
__device__ float luminanceWeight(float lum0, float lum1, float variance);

// Functions to sample and write G-buffer data
__device__ GBuffer sampleGBuffer(const IntegratorGlobals& globals, int2 c_pix);
__device__ void writeGBuffer(const IntegratorGlobals& globals, const GBuffer& gbuffer, int2 c_pix);

// Function that performs the SVGF pass
__device__ void psvgf(const IntegratorGlobals& globals, int2 c_pix, int stepSize);