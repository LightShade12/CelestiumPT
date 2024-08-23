#include "Renderer.hpp"
#include <cuda_gl_interop.h>
#include "kernel.hpp"
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

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

struct CudaAPI
{
	cudaGraphicsResource_t m_CompositeRenderTargetTextureCudaResource;
	dim3 m_BlockGridDimensions;
	dim3 m_ThreadBlockDimensions;
};

Renderer::Renderer()
{
	m_CudaResourceAPI = new CudaAPI();
	//TODO: redundant
	m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
	m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);
}

void Renderer::resizeResolution(int width, int height)
{
	if (width == m_NativeRenderResolutionWidth && height == m_NativeRenderResolutionHeight)return;
	m_NativeRenderResolutionHeight = height;
	m_NativeRenderResolutionWidth = width;

	if (m_CompositeRenderTargetTextureName != NULL) {
		m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
		m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);

		//m_AccumulationFrameBuffer->ColorDataBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);

		// unregister
		cudaGraphicsUnregisterResource(m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource);
		// resize
		glBindTexture(GL_TEXTURE_2D, m_CompositeRenderTargetTextureName);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// register back
		cudaGraphicsGLRegisterImage(&(m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource), m_CompositeRenderTargetTextureName, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	}
	else //Texture Creation
	{
		m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
		m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);
		//m_AccumulationFrameBuffer->ColorDataBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);

		//GL texture configure
		glGenTextures(1, &m_CompositeRenderTargetTextureName);
		glBindTexture(GL_TEXTURE_2D, m_CompositeRenderTargetTextureName);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		//TODO: make a switchable frame filtering mode
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaGraphicsGLRegisterImage(&(m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource), m_CompositeRenderTargetTextureName, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	}
}

void Renderer::renderFrame()
{
	cudaGraphicsMapResources(1, &(m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource));

	cudaArray_t render_target_texture_sub_resource_array;
	cudaGraphicsSubResourceGetMappedArray(&render_target_texture_sub_resource_array, m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource, 0, 0);
	cudaResourceDesc render_target_texture_resource_descriptor;
	{
		render_target_texture_resource_descriptor.resType = cudaResourceTypeArray;
		render_target_texture_resource_descriptor.res.array.array = render_target_texture_sub_resource_array;
	}
	cudaSurfaceObject_t render_target_texture_surface_object;
	cudaCreateSurfaceObject(&render_target_texture_surface_object, &render_target_texture_resource_descriptor);

	//----
	invokeKernel(render_target_texture_surface_object, m_NativeRenderResolutionWidth,
		m_NativeRenderResolutionHeight, m_CudaResourceAPI->m_BlockGridDimensions, m_CudaResourceAPI->m_ThreadBlockDimensions);
	//invokeRenderKernel(render_target_texture_surface_object, m_BufferWidth, m_BufferHeight,
		//m_BlockGridDimensions, m_ThreadBlockDimensions, m_CurrentCamera.getDeviceCamera(), m_DeviceSceneData, m_FrameIndex,
		//thrust::raw_pointer_cast(m_AccumulationFrameBuffer->ColorDataBuffer.data()));

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	//----

		//post render cuda---------------------------------------------------------------------------------
	cudaDestroySurfaceObject(render_target_texture_surface_object);
	cudaGraphicsUnmapResources(1, &(m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource));
	cudaStreamSynchronize(0);
	//m_FrameIndex++;
}

Renderer::~Renderer()
{
	delete m_CudaResourceAPI;
	glDeleteTextures(1, &m_CompositeRenderTargetTextureName);
}