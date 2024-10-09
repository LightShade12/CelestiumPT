#pragma once
#include "linear_algebra.cuh"

__device__ float balanceHeuristic(int nf, float fPdf, int ng, float gPdf);

__device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf);