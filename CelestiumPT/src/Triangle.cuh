#pragma once

#include "maths/vector_maths.cuh"
#include <glm/glm.hpp>

struct Vertex {
	Vertex(float3 p, float3 n, float2 uv) :position(p), normal(n), UV(uv) {}
	Vertex(glm::vec3 p, glm::vec3 n, glm::vec2 uv) :
		position(make_float3(p.x, p.y, p.z)),
		normal(make_float3(n.x, n.y, n.z)),
		UV(make_float2(uv.x, uv.y))
	{}
	float3 position;
	float3 normal;
	float2 UV;
};

struct ShapeSampleContext {
	__device__ ShapeSampleContext() = default;
	__device__ ShapeSampleContext(const float3& p, const float3& n, const float3& ns) : p(p), n(n), ns(ns) {}

	float3 p, n, ns;
};

struct ShapeSample {
	__device__ ShapeSample(const float3& p, const float3& n, float pdf) : p(p), n(n), pdf(pdf) {}

	float3 p;
	float3 n;
	float pdf = 0;
};

//TODO:lookup docs
__device__ float3 sampleUniformTriangle(float2 u);

struct Triangle {
	Triangle(Vertex v0, Vertex v1, Vertex v2, glm::vec3 nrm);

	__host__ __device__ float area();

	__device__ ShapeSample sample(const ShapeSampleContext& ctx, float2 u2);

	int LightIdx = -1;
	Vertex vertex0, vertex1, vertex2;
	float3 centroid{}; //TODO: move this out of triangle
	float3 face_normal{};
};