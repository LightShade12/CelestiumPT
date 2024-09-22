#pragma once

#include "vector_maths.cuh"

#include "glm/mat4x4.hpp"
#include <stdio.h>

// column-major maths
class Matrix3x3 {
public:
	__host__ __device__ Matrix3x3() : columns{ make_float3(1, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1) } {};

	__host__ __device__ explicit Matrix3x3(float val) {
		columns[0] = make_float3(val, 0, 0);
		columns[1] = make_float3(0, val, 0);
		columns[2] = make_float3(0, 0, val);
	};

	__host__ explicit Matrix3x3(const glm::mat3& mat)
	{
		columns[0] = make_float3(mat[0][0], mat[0][1], mat[0][2]);  // First column
		columns[1] = make_float3(mat[1][0], mat[1][1], mat[1][2]);  // Second column
		columns[2] = make_float3(mat[2][0], mat[2][1], mat[2][2]);  // Third column
	}

	__host__ __device__ Matrix3x3(float3 c1, float3 c2, float3 c3) {
		columns[0] = c1; columns[1] = c2; columns[2] = c3;
	};

	__host__ __device__ Matrix3x3(
		float c1r1, float c1r2, float c1r3,
		float c2r1, float c2r2, float c2r3,
		float c3r1, float c3r2, float c3r3)
	{
		columns[0] = make_float3(c1r1, c1r2, c1r3);
		columns[1] = make_float3(c2r1, c2r2, c2r3);
		columns[2] = make_float3(c3r1, c3r2, c3r3);
	};

	__host__ __device__ float3& operator [] (size_t idx) {
		return columns[idx];
	}
	__host__ __device__ const float3& operator[](size_t idx) const {
		return columns[idx];
	}

	__host__ __device__ Matrix3x3 transpose() const {
		return Matrix3x3(
			make_float3(columns[0].x, columns[1].x, columns[2].x),
			make_float3(columns[0].y, columns[1].y, columns[2].y),
			make_float3(columns[0].z, columns[1].z, columns[2].z)
		);
	}

	__host__ __device__ Matrix3x3 inverse() const {
		// Compute the determinant
		const float3& c0 = columns[0];
		const float3& c1 = columns[1];
		const float3& c2 = columns[2];

		float det = c0.x * (c1.y * c2.z - c1.z * c2.y) -
			c1.x * (c0.y * c2.z - c0.z * c2.y) +
			c2.x * (c0.y * c1.z - c0.z * c1.y);

		if (fabs(det) < 1e-6f) {
			return Matrix3x3::getIdentity();  // Not invertible, return identity (or handle differently)
		}

		float invDet = 1.0f / det;

		Matrix3x3 inv;

		// Compute the inverse matrix elements
		inv[0].x = (c1.y * c2.z - c1.z * c2.y) * invDet;
		inv[0].y = (c0.z * c2.y - c0.y * c2.z) * invDet;
		inv[0].z = (c0.y * c1.z - c0.z * c1.y) * invDet;

		inv[1].x = (c1.z * c2.x - c1.x * c2.z) * invDet;
		inv[1].y = (c0.x * c2.z - c0.z * c2.x) * invDet;
		inv[1].z = (c0.z * c1.x - c0.x * c1.z) * invDet;

		inv[2].x = (c1.x * c2.y - c1.y * c2.x) * invDet;
		inv[2].y = (c0.y * c2.x - c0.x * c2.y) * invDet;
		inv[2].z = (c0.x * c1.y - c0.y * c1.x) * invDet;

		return inv;
	}

	__host__ __device__ float3 operator*(const float3& vec) const {
		float3 result;

		result.x = columns[0].x * vec.x + columns[1].x * vec.y + columns[2].x * vec.z;
		result.y = columns[0].y * vec.x + columns[1].y * vec.y + columns[2].y * vec.z;
		result.z = columns[0].z * vec.x + columns[1].z * vec.y + columns[2].z * vec.z;

		return result;
	}

	__host__ __device__ Matrix3x3 operator*(const Matrix3x3& mat2) const {
		Matrix3x3 res;
		Matrix3x3 mat1 = this->transpose();  // rows enabled; interpret as columns

		// Multiply each row with all columns; fill up entire rows
		for (int rowidx = 0; rowidx < 3; rowidx++) {
			res[rowidx].x = dot(mat1[rowidx], mat2[0]);
			res[rowidx].y = dot(mat1[rowidx], mat2[1]);
			res[rowidx].z = dot(mat1[rowidx], mat2[2]);
		}

		return res.transpose();  // rows back to columns
	}

	__host__ __device__ static Matrix3x3 getIdentity() {
		return Matrix3x3(
			make_float3(1, 0, 0),
			make_float3(0, 1, 0),
			make_float3(0, 0, 1)
		);
	}

	__host__ __device__ static Matrix3x3 getZero() {
		return Matrix3x3(
			make_float3(0, 0, 0),
			make_float3(0, 0, 0),
			make_float3(0, 0, 0)
		);
	}

	__host__ glm::mat3 toGLM() {
		return glm::mat3(
			columns[0].x, columns[0].y, columns[0].z,
			columns[1].x, columns[1].y, columns[1].z,
			columns[2].x, columns[2].y, columns[2].z
		);
	}

	__host__ __device__ static void print_matrix(const Matrix3x3& mat) {
		printf("\n");
		printf("| %.3f %.3f %.3f |\n", mat[0].x, mat[1].x, mat[2].x);
		printf("| %.3f %.3f %.3f |\n", mat[0].y, mat[1].y, mat[2].y);
		printf("| %.3f %.3f %.3f |\n", mat[0].z, mat[1].z, mat[2].z);
	}

private:
	float3 columns[3] = {};
};

//column major maths
class Matrix4x4 {
public:
	__host__ __device__ Matrix4x4() : columns{ make_float4(1, 0, 0,0), make_float4(0, 1, 0,0),
		make_float4(0, 0, 1,0),make_float4(0,0,0,1) } {};

	__host__ __device__ explicit Matrix4x4(float val) {
		columns[0] = make_float4(val, 0, 0, 0);
		columns[1] = make_float4(0, val, 0, 0);
		columns[2] = make_float4(0, 0, val, 0);
		columns[3] = make_float4(0, 0, 0, val);
	};

	__host__ explicit Matrix4x4(const glm::mat4& mat)
	{
		columns[0] = make_float4(mat[0][0], mat[0][1], mat[0][2], mat[0][3]);  // First column
		columns[1] = make_float4(mat[1][0], mat[1][1], mat[1][2], mat[1][3]);  // Second column
		columns[2] = make_float4(mat[2][0], mat[2][1], mat[2][2], mat[2][3]);  // Third column
		columns[3] = make_float4(mat[3][0], mat[3][1], mat[3][2], mat[3][3]);  // Fourth column
	}

	__host__ __device__ Matrix4x4(float4 c1, float4 c2, float4 c3, float4 c4) {
		columns[0] = c1; columns[1] = c2; columns[2] = c3; columns[3] = c4;
	};

	__host__ __device__ Matrix4x4(
		float c1r1, float c1r2, float c1r3, float c1r4,
		float c2r1, float c2r2, float c2r3, float c2r4,
		float c3r1, float c3r2, float c3r3, float c3r4,
		float c4r1, float c4r2, float c4r3, float c4r4)
	{
		columns[0] = make_float4(c1r1, c1r2, c1r3, c1r4);
		columns[1] = make_float4(c2r1, c2r2, c2r3, c2r4);
		columns[2] = make_float4(c3r1, c3r2, c3r3, c3r4);
		columns[3] = make_float4(c4r1, c4r2, c4r3, c4r4);
	};

	__host__ __device__ float4& operator [] (size_t idx) {
		return columns[idx];
	}
	__host__ __device__ const float4& operator[](size_t idx) const {
		return columns[idx];
	}

	__host__ __device__ Matrix4x4 inverse() const {
		// Alias for readability
		const float4& c0 = columns[0];
		const float4& c1 = columns[1];
		const float4& c2 = columns[2];
		const float4& c3 = columns[3];

		// Compute the minors for the determinant calculation
		float subfactor00 = c2.z * c3.w - c3.z * c2.w;
		float subfactor01 = c2.y * c3.w - c3.y * c2.w;
		float subfactor02 = c2.y * c3.z - c3.y * c2.z;
		float subfactor03 = c2.x * c3.w - c3.x * c2.w;
		float subfactor04 = c2.x * c3.z - c3.x * c2.z;
		float subfactor05 = c2.x * c3.y - c3.x * c2.y;

		// Compute the determinant
		float det = c0.x * (c1.y * subfactor00 - c1.z * subfactor01 + c1.w * subfactor02) -
			c0.y * (c1.x * subfactor00 - c1.z * subfactor03 + c1.w * subfactor04) +
			c0.z * (c1.x * subfactor01 - c1.y * subfactor03 + c1.w * subfactor05) -
			c0.w * (c1.x * subfactor02 - c1.y * subfactor04 + c1.z * subfactor05);

		if (fabs(det) < 1e-6f) {
			// Matrix is not invertible, return the identity matrix (or handle error)
			return Matrix4x4::getIdentity();
		}

		// Inverse is 1/det * adjugate
		float invDet = 1.0f / det;

		Matrix4x4 inverseMatrix;

		// Compute each element of the adjugate (transposed cofactor matrix) multiplied by invDet
		inverseMatrix[0].x = (c1.y * subfactor00 - c1.z * subfactor01 + c1.w * subfactor02) * invDet;
		inverseMatrix[0].y = -(c0.y * subfactor00 - c0.z * subfactor01 + c0.w * subfactor02) * invDet;
		inverseMatrix[0].z = (c0.y * (c1.z * c3.w - c1.w * c3.z) - c0.z * (c1.y * c3.w - c1.w * c3.y) + c0.w * (c1.y * c3.z - c1.z * c3.y)) * invDet;
		inverseMatrix[0].w = -(c0.y * (c1.z * c2.w - c1.w * c2.z) - c0.z * (c1.y * c2.w - c1.w * c2.y) + c0.w * (c1.y * c2.z - c1.z * c2.y)) * invDet;

		inverseMatrix[1].x = -(c1.x * subfactor00 - c1.z * subfactor03 + c1.w * subfactor04) * invDet;
		inverseMatrix[1].y = (c0.x * subfactor00 - c0.z * subfactor03 + c0.w * subfactor04) * invDet;
		inverseMatrix[1].z = -(c0.x * (c1.z * c3.w - c1.w * c3.z) - c0.z * (c1.x * c3.w - c1.w * c3.x) + c0.w * (c1.x * c3.z - c1.z * c3.x)) * invDet;
		inverseMatrix[1].w = (c0.x * (c1.z * c2.w - c1.w * c2.z) - c0.z * (c1.x * c2.w - c1.w * c2.x) + c0.w * (c1.x * c2.z - c1.z * c2.x)) * invDet;

		inverseMatrix[2].x = (c1.x * subfactor01 - c1.y * subfactor03 + c1.w * subfactor05) * invDet;
		inverseMatrix[2].y = -(c0.x * subfactor01 - c0.y * subfactor03 + c0.w * subfactor05) * invDet;
		inverseMatrix[2].z = (c0.x * (c1.y * c3.w - c1.w * c3.y) - c0.y * (c1.x * c3.w - c1.w * c3.x) + c0.w * (c1.x * c3.y - c1.y * c3.x)) * invDet;
		inverseMatrix[2].w = -(c0.x * (c1.y * c2.w - c1.w * c2.y) - c0.y * (c1.x * c2.w - c1.w * c2.x) + c0.w * (c1.x * c2.y - c1.y * c2.x)) * invDet;

		inverseMatrix[3].x = -(c1.x * subfactor02 - c1.y * subfactor04 + c1.z * subfactor05) * invDet;
		inverseMatrix[3].y = (c0.x * subfactor02 - c0.y * subfactor04 + c0.z * subfactor05) * invDet;
		inverseMatrix[3].z = -(c0.x * (c1.y * c3.z - c1.z * c3.y) - c0.y * (c1.x * c3.z - c1.z * c3.x) + c0.z * (c1.x * c3.y - c1.y * c3.x)) * invDet;
		inverseMatrix[3].w = (c0.x * (c1.y * c2.z - c1.z * c2.y) - c0.y * (c1.x * c2.z - c1.z * c2.x) + c0.z * (c1.x * c2.y - c1.y * c2.x)) * invDet;

		return inverseMatrix;
	}

	__host__ __device__ float4 operator*(const float4& vec) const {
		float4 result;

		result.x = columns[0].x * vec.x + columns[1].x * vec.y + columns[2].x * vec.z + columns[3].x * vec.w;
		result.y = columns[0].y * vec.x + columns[1].y * vec.y + columns[2].y * vec.z + columns[3].y * vec.w;
		result.z = columns[0].z * vec.x + columns[1].z * vec.y + columns[2].z * vec.z + columns[3].z * vec.w;
		result.w = columns[0].w * vec.x + columns[1].w * vec.y + columns[2].w * vec.z + columns[3].w * vec.w;

		return result;
	}

	//caller is left side matrix
	__host__ __device__ Matrix4x4 operator*(const Matrix4x4& mat2) const {
		Matrix4x4 res;//intepreted as row major storage
		Matrix4x4 mat1 = this->transpose();//rows enabled;interpret as columns

		//mult each row with all columns; fillup entire rows
		for (int rowidx = 0; rowidx < 4; rowidx++) {
			res[rowidx].x = dot(mat1[rowidx], mat2[0]);
			res[rowidx].y = dot(mat1[rowidx], mat2[1]);
			res[rowidx].z = dot(mat1[rowidx], mat2[2]);
			res[rowidx].w = dot(mat1[rowidx], mat2[3]);
		}

		return res.transpose();//rows back to columns
	}

	__host__ __device__ Matrix4x4 transpose() const {
		return Matrix4x4(
			make_float4(columns[0].x, columns[1].x, columns[2].x, columns[3].x),
			make_float4(columns[0].y, columns[1].y, columns[2].y, columns[3].y),
			make_float4(columns[0].z, columns[1].z, columns[2].z, columns[3].z),
			make_float4(columns[0].w, columns[1].w, columns[2].w, columns[3].w)
		);
	}

	__host__ __device__ static Matrix4x4 getIdentity() {
		return Matrix4x4(
			make_float4(1, 0, 0, 0),
			make_float4(0, 1, 0, 0),
			make_float4(0, 0, 1, 0),
			make_float4(0, 0, 0, 1)
		);
	}

	__host__ __device__ static Matrix4x4 getZero() {
		return Matrix4x4(
			make_float4(0, 0, 0, 0),
			make_float4(0, 0, 0, 0),
			make_float4(0, 0, 0, 0),
			make_float4(0, 0, 0, 0)
		);
	}

	__host__ glm::mat4 toGLM() {
		return glm::mat4(
			columns[0].x, columns[0].y, columns[0].z, columns[0].w,
			columns[1].x, columns[1].y, columns[1].z, columns[1].w,
			columns[2].x, columns[2].y, columns[2].z, columns[2].w,
			columns[3].x, columns[3].y, columns[3].z, columns[3].w
		);
	}

	__host__ __device__ static void print_matrix(const Matrix4x4& mat) {
		printf("\n");
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].x, mat[1].x, mat[2].x, mat[3].x);
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].y, mat[1].y, mat[2].y, mat[3].y);
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].z, mat[1].z, mat[2].z, mat[3].z);
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].w, mat[1].w, mat[2].w, mat[3].w);
	}
private:
	float4 columns[4] = {};
};

//column major maths
using Mat4 = Matrix4x4;
using Mat3 = Matrix3x3;