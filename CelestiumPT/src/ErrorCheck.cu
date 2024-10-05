#include "ErrorCheck.cuh"
#include <iostream>

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

const char* glErrorString(GLenum err)
{
	switch (err) {
	case GL_INVALID_ENUM: return "Invalid Enum";
	case GL_INVALID_VALUE: return "Invalid Value";
	case GL_INVALID_OPERATION: return "Invalid Operation";
	case GL_STACK_OVERFLOW: return "Stack Overflow";
	case GL_STACK_UNDERFLOW: return "Stack Underflow";
	case GL_OUT_OF_MEMORY: return "Out of Memory";
		//case GL_TABLE_TOO_LARGE: return "Table too Large";
	default: return "Unknown Error";
	}
}

const char* fboErrorString(GLenum status)
{
	switch (status) {
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: return "Incomplete Attachment";
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return "Missing Attachment";
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: return "Incomplete Draw Buffer";
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: return "Incomplete Read Buffer";
	case GL_FRAMEBUFFER_UNSUPPORTED: return "Unsupposed Configuration";
	default: return "Unknown Error";
	}
}
