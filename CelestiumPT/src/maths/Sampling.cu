#include "Sampling.cuh"

__device__ float balanceHeuristic(int nf, float fPdf, int ng, float gPdf)
{
	return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

__device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return Sqr(f) / (Sqr(f) + Sqr(g));
}