#pragma once
#include <glm/gtc/matrix_transform.hpp>

static void print_matrix(const glm::mat4& mat) {
	printf("\n");
	printf("| %.3f %.3f %.3f %.3f |\n", mat[0].x, mat[1].x, mat[2].x, mat[3].x);
	printf("| %.3f %.3f %.3f %.3f |\n", mat[0].y, mat[1].y, mat[2].y, mat[3].y);
	printf("| %.3f %.3f %.3f %.3f |\n", mat[0].z, mat[1].z, mat[2].z, mat[3].z);
	printf("| %.3f %.3f %.3f %.3f |\n\n\n", mat[0].w, mat[1].w, mat[2].w, mat[3].w);
}