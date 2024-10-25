#pragma once

#include <stdexcept>
#include <glad/glad.h>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

const char* glErrorString(GLenum err);

const char* fboErrorString(GLenum status);

class gl_error : public std::runtime_error {
public:
	gl_error(std::string location, GLenum err) : std::runtime_error(glErrorString(err) + location) {}
};

class fbo_error : public std::runtime_error {
public:
	fbo_error(std::string location, GLenum status) : std::runtime_error(fboErrorString(status) + location) {}
};

#define STRING(X) #X
#define TOSTRING(X) STRING(X)
#define FILE_LINE " @ " __FILE__ ":" TOSTRING(__LINE__)
#ifndef NDEBUG
#define GL_CHECK(func) func; { GLenum glerr = glGetError(); if(GL_NO_ERROR != glerr) throw gl_error(FILE_LINE, glerr); }
#define FBO_CHECK { GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER); if(GL_FRAMEBUFFER_COMPLETE != status) throw fbo_error(FILE_LINE, status); }
#else
#define GL_CHECK(func) func;
#define FBO_CHECK
#endif