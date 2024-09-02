#include "Renderer.hpp"
//#include "HostCamera.hpp"
#include "ErrorCheck.cuh"
#include <cuda_gl_interop.h>
#include "Integrator.cuh"

#include "SceneGeometry.cuh"
#include "DeviceScene.cuh"
//#include "Triangle.cuh"

#include <iostream>
/*
* Mostly adapter and inter conversion work
*/

struct CudaAPI
{
	cudaGraphicsResource_t m_CompositeRenderTargetTextureCudaResource;
	dim3 m_BlockGridDimensions;
	dim3 m_ThreadBlockDimensions;
};

struct CelestiumPT_API
{
	DeviceScene DeviceScene;
	IntegratorGlobals m_IntegratorGlobals;
};

Renderer::Renderer()
{
	m_CudaResourceAPI = new CudaAPI();
	m_CelestiumPTResourceAPI = new CelestiumPT_API();
	//TODO: redundant
	m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
	m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);

	cudaMallocManaged(&m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_camera, sizeof(DeviceCamera));
	cudaMallocManaged(&m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_aggregate, sizeof(SceneGeometry));

	m_CurrentCamera = HostCamera(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_camera);
	m_CelestiumPTResourceAPI->DeviceScene = DeviceScene(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_aggregate);
	m_CurrentScene = HostScene(&(m_CelestiumPTResourceAPI->DeviceScene));

	//m_CurrentScene.AddTriangle(
	//	{ 0,3.75,-10 }, { 0,0,1 },
	//	{ 3,-1.0,-10 }, { 0,0,1 },
	//	{ -3,-1.0,-10 }, { 0,0,1 },
	//	{ 0,0,1 });
	//
	//m_CurrentScene.AddTriangle(
	//	{ 0,-1.75,-7 }, { 0,0,1 },
	//	{ 3,-8.0,-7 }, { 0,0,1 },
	//	{ -3,-8.0,-7 }, { 0,0,1 },
	//	{ 0,0,1 });

	//TODO: temporary; make this part of initing a camera
	m_CurrentCamera.setTransform(
		glm::mat4(
			glm::vec4(1, 0, 0, 0),
			glm::vec4(0, 1, 0, 0),
			glm::vec4(0, 0, -1, 0),
			glm::vec4(0, 0.5, 1.5, 0)
		)
	);
	m_CurrentCamera.setVectors(
		glm::vec4(0, 0.5, 1.5, 0),
		glm::vec4(0, 0, -1, 0),
		glm::vec4(0, 1, 0, 0),
		glm::vec4(1, 0, 0, 0)
	);
	m_CurrentCamera.updateDevice();
}

void Renderer::resizeResolution(int width, int height)
{
	if (width == m_NativeRenderResolutionWidth && height == m_NativeRenderResolutionHeight)return;
	m_NativeRenderResolutionHeight = height;
	m_NativeRenderResolutionWidth = width;

	if (m_CompositeRenderTargetTextureName == NULL)//Texture Creation
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
	else
	{
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
}

static uint32_t g_frameIndex = 0;

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

	//prepare globals--------------------
	cudaCreateSurfaceObject(&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_render_surface_object), &render_target_texture_resource_descriptor);
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.frameidx = g_frameIndex;
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.resolution = make_int2(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	IntegratorPipeline::invokeRenderKernel(m_CelestiumPTResourceAPI->m_IntegratorGlobals,
		m_CudaResourceAPI->m_BlockGridDimensions, m_CudaResourceAPI->m_ThreadBlockDimensions);
	g_frameIndex++;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//----

	//post render cuda---------------------------------------------------------------------------------
	cudaDestroySurfaceObject(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_render_surface_object);
	cudaGraphicsUnmapResources(1, &(m_CudaResourceAPI->m_CompositeRenderTargetTextureCudaResource));
	cudaStreamSynchronize(0);
	//m_FrameIndex++;
}

Renderer::~Renderer()
{
	cudaFree(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_camera);
	cudaFree(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_aggregate);
	delete m_CudaResourceAPI;
	delete m_CelestiumPTResourceAPI;
	glDeleteTextures(1, &m_CompositeRenderTargetTextureName);
}