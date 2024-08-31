#pragma once
#include <vector_types.h>

struct Vertex {
	float3 position;
	float3 normal;
};
struct Triangle {
	Vertex vertex0, vertex1, vertex2;
	float3 face_normal{};
};