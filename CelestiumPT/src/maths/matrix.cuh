#pragma once
#include "vector_maths.cuh"
#include <stdio.h>

//column major maths
class Matrix3x3 {
public:
	// Default constructor initializes to the identity matrix
	__device__ Matrix3x3() : columns{ make_float3(1, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1) } {}

	// Constructor with individual columns
	__device__ Matrix3x3(float3 c1, float3 c2, float3 c3) {
		columns[0] = c1; columns[1] = c2; columns[2] = c3;
	}

	// Accessor for non-const objects
	__device__ float3& operator [] (size_t idx) {
		return columns[idx];
	}

	// Accessor for const objects
	__device__ const float3& operator[](size_t idx) const {
		return columns[idx];
	}

	// Matrix-vector multiplication
	__device__ float3 operator*(const float3& vec) const {
		float3 result;
		result = columns[0] * vec.x + columns[1] * vec.y + columns[3] * vec.z;
		return result;
	}

	// Matrix-matrix multiplication
	__device__ Matrix3x3 operator*(const Matrix3x3& mat2) const {
		Matrix3x3 res; // Interpreted as row-major storage
		Matrix3x3 mat1 = this->transpose(); // Rows enabled; interpret as columns

		// Multiply each row with all columns; fill up entire rows
		for (int rowidx = 0; rowidx < 3; rowidx++) {
			res[rowidx].x = dot(mat1[rowidx], mat2[0]);
			res[rowidx].y = dot(mat1[rowidx], mat2[1]);
			res[rowidx].z = dot(mat1[rowidx], mat2[2]);
		}

		return res.transpose(); // Rows back to columns
	}

	// Transpose function
	__device__ Matrix3x3 transpose() const {
		return Matrix3x3(
			make_float3(columns[0].x, columns[1].x, columns[2].x),
			make_float3(columns[0].y, columns[1].y, columns[2].y),
			make_float3(columns[0].z, columns[1].z, columns[2].z)
		);
	}

	// Identity matrix generator
	__device__ static Matrix3x3 getIdentity() {
		return Matrix3x3(
			make_float3(1, 0, 0),
			make_float3(0, 1, 0),
			make_float3(0, 0, 1)
		);
	}

	// Zero matrix generator
	__device__ static Matrix3x3 getZero() {
		return Matrix3x3(
			make_float3(0, 0, 0),
			make_float3(0, 0, 0),
			make_float3(0, 0, 0)
		);
	}

	// Print matrix function (similar to Matrix4x4)
	__device__ static void print_matrix(const Matrix3x3& mat) {
		printf("| %.3f %.3f %.3f |\n", mat[0].x, mat[1].x, mat[2].x);
		printf("| %.3f %.3f %.3f |\n", mat[0].y, mat[1].y, mat[2].y);
		printf("| %.3f %.3f %.3f |\n", mat[0].z, mat[1].z, mat[2].z);
	}

private:
	// Columns storing the matrix data in column-major order
	float3 columns[3] = {};
};

//column major maths
class Matrix4x4 {
public:
	__host__ __device__ Matrix4x4() : columns{ make_float4(1, 0, 0,0), make_float4(0, 1, 0,0),
		make_float4(0, 0, 1,0),make_float4(0,0,0,1) } {};

	__host__ __device__ Matrix4x4(float val) {
		columns[0] = make_float4(val, 0, 0, 0);
		columns[1] = make_float4(0, val, 0, 0);
		columns[2] = make_float4(0, 0, val, 0);
		columns[3] = make_float4(0, 0, 0, val);
	};

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

	__host__ __device__ float3 operator*(const float3& vec) const {
		float3 result;

		result.x = columns[0].x * vec.x + columns[1].x * vec.y + columns[2].x * vec.z + columns[3].x;
		result.y = columns[0].y * vec.x + columns[1].y * vec.y + columns[2].y * vec.z + columns[3].y;
		result.z = columns[0].z * vec.x + columns[1].z * vec.y + columns[2].z * vec.z + columns[3].z;

		return result;
	}

	__host__ __device__ float3 operator*(const float4& vec) const {
		float3 result;

		result.x = columns[0].x * vec.x + columns[1].x * vec.y + columns[2].x * vec.z + columns[3].x * vec.w;
		result.y = columns[0].y * vec.x + columns[1].y * vec.y + columns[2].y * vec.z + columns[3].y * vec.w;
		result.z = columns[0].z * vec.x + columns[1].z * vec.y + columns[2].z * vec.z + columns[3].z * vec.w;

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

	__host__ __device__ static void print_matrix(const Matrix4x4& mat) {
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].x, mat[1].x, mat[2].x, mat[3].x);
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].y, mat[1].y, mat[2].y, mat[3].y);
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].z, mat[1].z, mat[2].z, mat[3].z);
		printf("| %.3f %.3f %.3f %.3f |\n", mat[0].w, mat[1].w, mat[2].w, mat[3].w);
	}
private:
	float4 columns[4] = {};
};

//column major maths
using Mat3 = Matrix3x3;
using Mat4 = Matrix4x4;