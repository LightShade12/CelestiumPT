#include "ShapeIntersection.cuh"
#include "BSDF.cuh"
#include "Ray.cuh"
#include "Spectrum.cuh"
#include "SceneGeometry.cuh"
#include "Storage.cuh"

__device__ float3 getGeometricTangent(const Triangle& triangle) {
	float3 edge0 = triangle.vertex1.position - triangle.vertex0.position;
	float3 edge1 = triangle.vertex2.position - triangle.vertex0.position;
	float2 deltaUV0 = triangle.vertex1.UV - triangle.vertex0.UV;
	float2 deltaUV1 = triangle.vertex2.UV - triangle.vertex0.UV;
	float invDet = 1.0f / (deltaUV0.x * deltaUV1.y - deltaUV1.x * deltaUV0.y);
	float3 tangent = invDet * (deltaUV1.y * edge0 - deltaUV0.y * edge1);
	return normalize(tangent);
}

//assumes orthogonality
__device__ Mat3 getTBNMatrix(float3 ns, const Triangle& triangle)
{
	float3 tan_geo = getGeometricTangent(triangle);
	float3 bitan = cross(ns, tan_geo);
	float3 tan_s = cross(bitan, ns);
	Mat3 TBN(tan_s, normalize(bitan), ns);
	return TBN;
}

__device__ BSDF ShapeIntersection::getBSDF(const IntegratorGlobals& globals)
{
	const Triangle& triangle = globals.SceneDescriptor.device_geometry_aggregate->DeviceTrianglesBuffer[triangle_idx];
	return BSDF(getTBNMatrix(w_shading_norm, triangle));
}

__device__ RGBSpectrum ShapeIntersection::Le(float3 w)
{
	//if(arealight!=nullptr)printf("light scale: %.3f | ", arealight->scale);
	return (arealight != nullptr) ?
		arealight->L(w_pos, w_geo_norm, w) :
		make_float3(0);
}

__device__ Ray ShapeIntersection::spawnRay(const float3& wi)
{
	float3 orig = w_pos + (w_geo_norm * HIT_EPSILON);
	return Ray(orig, wi);
}