#pragma once
#include <vector_types.h>
#include <glm/glm.hpp>

struct Vertex {
	Vertex(float3 p, float3 n) :position(p), normal(n) {}
	Vertex(glm::vec3 p, glm::vec3 n) :position(make_float3(p.x, p.y, p.z)), normal(make_float3(n.x, n.y, n.z)) {}
	float3 position;
	float3 normal;
};
struct Triangle {
	Triangle(Vertex v0, Vertex v1, Vertex v2, glm::vec3 nrm) :vertex0(v0), vertex1(v1), vertex2(v2),
		face_normal(make_float3(nrm.x, nrm.y, nrm.z)) {};
	Vertex vertex0, vertex1, vertex2;
	float3 face_normal{};
};