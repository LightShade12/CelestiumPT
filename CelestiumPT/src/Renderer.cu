#include "Renderer.hpp"
#include "ErrorCheck.cuh"
#include "Storage.cuh"
#include "Integrator.cuh"

#include "SceneGeometry.cuh"
#include "DeviceScene.cuh"
#include "DeviceCamera.cuh"
//#include "Triangle.cuh"

#include <cuda_gl_interop.h>
#include <iostream>
/*
* Mostly adapter and inter conversion work
*/

class FrameBuffer {
public:

	FrameBuffer() = default;

	void initialize(int width, int height)
	{
		//GL texture configure
		glGenTextures(1, &m_RenderTargetTextureName);
		glBindTexture(GL_TEXTURE_2D, m_RenderTargetTextureName);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		//TODO: make a switchable frame filtering mode
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);

		cudaGraphicsGLRegisterImage(&m_RenderTargetTextureCudaResource, m_RenderTargetTextureName, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	};

	~FrameBuffer() {
		glDeleteTextures(1, &m_RenderTargetTextureName);
	}

	void resizeResolution(int width, int height) {
		if (m_RenderTargetTextureName == NULL) {
			initialize(width, height); return;
		}
		// unregister
		cudaGraphicsUnregisterResource(m_RenderTargetTextureCudaResource);
		// resize
		glBindTexture(GL_TEXTURE_2D, m_RenderTargetTextureName);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		// register back
		cudaGraphicsGLRegisterImage(&m_RenderTargetTextureCudaResource, m_RenderTargetTextureName, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	};

	void beginRender(cudaSurfaceObject_t* surf_obj) {
		cudaGraphicsMapResources(1, &m_RenderTargetTextureCudaResource);

		cudaArray_t render_target_texture_sub_resource_array;
		cudaGraphicsSubResourceGetMappedArray(&render_target_texture_sub_resource_array, m_RenderTargetTextureCudaResource, 0, 0);
		cudaResourceDesc render_target_texture_resource_descriptor;
		{
			render_target_texture_resource_descriptor.resType = cudaResourceTypeArray;
			render_target_texture_resource_descriptor.res.array.array = render_target_texture_sub_resource_array;
		}

		//prepare globals--------------------
		cudaCreateSurfaceObject(surf_obj, &render_target_texture_resource_descriptor);
	}

	void endRender(cudaSurfaceObject_t* surf_obj)
	{
		cudaDestroySurfaceObject(*surf_obj);
		cudaGraphicsUnmapResources(1, &m_RenderTargetTextureCudaResource);
		cudaStreamSynchronize(0);
	}

	GLuint m_RenderTargetTextureName = NULL;
	cudaGraphicsResource_t m_RenderTargetTextureCudaResource;
};

struct CudaAPI
{
	//cudaGraphicsResource_t m_CompositeRenderTargetTextureCudaResource;
	dim3 m_BlockGridDimensions;
	dim3 m_ThreadBlockDimensions;
};

struct CelestiumPT_API
{
	DeviceScene DeviceScene;
	IntegratorGlobals m_IntegratorGlobals;
	FrameBuffer CompositeRenderBuffer;
	FrameBuffer NormalsRenderBuffer;
	FrameBuffer PositionsRenderBuffer;
	thrust::device_vector<float3>AccumulationFrameBuffer;
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

	//TODO: temporary; make this part of initing a camera
	m_CurrentCamera.setTransform(
		glm::mat4(
			glm::vec4(1, 0, 0, 0),
			glm::vec4(0, 1, 0, 0),
			glm::vec4(0, 0, -1, 0),
			glm::vec4(0, 0.5, 5.5, 0)
		)
	);

	m_CurrentCamera.updateDevice();
}

void Renderer::resizeResolution(int width, int height)
{
	if (width == m_NativeRenderResolutionWidth && height == m_NativeRenderResolutionHeight)return;
	m_NativeRenderResolutionHeight = height;
	m_NativeRenderResolutionWidth = width;

	m_CelestiumPTResourceAPI->CompositeRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->NormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->PositionsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
	m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);

	m_CelestiumPTResourceAPI->AccumulationFrameBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);
}

static uint32_t g_frameIndex = 1;

void Renderer::renderFrame()
{
	//pre render-------------------------------------------------------------

	m_CelestiumPTResourceAPI->CompositeRenderBuffer.beginRender(
		&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_render_surface_object));
	m_CelestiumPTResourceAPI->NormalsRenderBuffer.beginRender(
		&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.normals_render_surface_object));
	m_CelestiumPTResourceAPI->PositionsRenderBuffer.beginRender(
		&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.positions_render_surface_object));

	//prepare globals--------------------
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.frameidx = g_frameIndex;
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.resolution =
		make_int2(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.accumulation_framebuffer =
		thrust::raw_pointer_cast(m_CelestiumPTResourceAPI->AccumulationFrameBuffer.data());

	IntegratorPipeline::invokeRenderKernel(m_CelestiumPTResourceAPI->m_IntegratorGlobals,
		m_CudaResourceAPI->m_BlockGridDimensions, m_CudaResourceAPI->m_ThreadBlockDimensions);
	g_frameIndex++;

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//----

	//post render cuda---------------------------------------------------------------------------------
	m_CelestiumPTResourceAPI->CompositeRenderBuffer.endRender(
		&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_render_surface_object));
	m_CelestiumPTResourceAPI->NormalsRenderBuffer.endRender(
		&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.normals_render_surface_object));
	m_CelestiumPTResourceAPI->PositionsRenderBuffer.endRender(
		&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.positions_render_surface_object));
}

void Renderer::clearAccumulation()
{
	thrust::fill(m_CelestiumPTResourceAPI->AccumulationFrameBuffer.begin(),
		m_CelestiumPTResourceAPI->AccumulationFrameBuffer.end(), make_float3(0, 0, 0));
	g_frameIndex = 1;
}

GLuint Renderer::getCompositeRenderTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->CompositeRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getNormalsTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->NormalsRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getPositionsTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->PositionsRenderBuffer.m_RenderTargetTextureName;
}

Renderer::~Renderer()
{
	cudaFree(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_camera);
	cudaFree(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.dev_aggregate);
	delete m_CudaResourceAPI;
	delete m_CelestiumPTResourceAPI;
}