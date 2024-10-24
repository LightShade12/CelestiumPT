#pragma once
#include "storage.cuh"
#include <cuda_runtime.h>

// Temporal integration kernel for SVGF; out: filtered_irradiance frontbuff; feedbacks only moments to backbuff
__global__ void temporalAccumulate(const IntegratorGlobals globals);