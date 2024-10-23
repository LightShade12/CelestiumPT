#include "renderer.hpp"
#include "error_check.cuh"
#include "storage.cuh"
#include "integrator.cuh"
#include "svgf/svgf_passes.cuh"
#include "svgf/svgf_temporal_reprojection.cuh"
#include "render_passes.cuh"
#include "temporal_pass.cuh"

#include "scene_geometry.cuh"
#include "device_scene.cuh"
#include "device_camera.cuh"
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
		cudaGraphicsSubResourceGetMappedArray(&render_target_texture_sub_resource_array, m_RenderTargetTextureCudaResource,
			0, 0);
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
	dim3 m_BlockGridDimensions;
	dim3 m_ThreadBlockDimensions;
};

struct CelestiumPT_API
{
	DeviceScene DeviceScene;
	IntegratorGlobals m_IntegratorGlobals;

	FrameBuffer CompositeRenderBuffer;//srgb view transformed

	FrameBuffer IrradianceRenderBuffer;
	FrameBuffer MomentsRenderBuffer;
	FrameBuffer FilteredVarianceRenderFrontBuffer;
	FrameBuffer FilteredVarianceRenderBackBuffer;
	FrameBuffer FilteredIrradianceFrontBuffer;
	FrameBuffer FilteredIrradianceBackBuffer;

	FrameBuffer AlbedoRenderBuffer;
	FrameBuffer LocalNormalsRenderBuffer;//used for normals reject
	FrameBuffer DepthRenderBuffer;//used for rejection
	FrameBuffer LocalPositionsRenderBuffer;//used for reproj & depth reject
	FrameBuffer ObjectIDRenderBuffer; //used for reproj & reject

	FrameBuffer VelocityRenderBuffer;//used for reproj

	//history
	FrameBuffer HistoryIntegratedIrradianceRenderFrontBuffer;//read
	FrameBuffer HistoryIntegratedIrradianceRenderBackBuffer;//write; Filtered output
	FrameBuffer HistoryIntegratedMomentsFrontBuffer;
	FrameBuffer HistoryIntegratedMomentsBackBuffer;
	FrameBuffer HistoryDepthRenderBuffer;
	FrameBuffer HistoryWorldNormalsRenderBuffer;

	//DEBUG
	FrameBuffer WorldNormalsRenderBuffer;//debug
	FrameBuffer PositionsRenderBuffer;//debug
	FrameBuffer ObjectIDDebugRenderBuffer;
	FrameBuffer GASDebugRenderBuffer;
	FrameBuffer HitHeatmapDebugRenderBuffer;
	FrameBuffer BboxHeatmapDebugRenderBuffer;
	FrameBuffer UVsDebugRenderBuffer;
	FrameBuffer BarycentricsDebugRenderBuffer;

	thrust::device_vector<float3>AccumulationFrameBuffer;
	thrust::device_vector<float3>VarianceAccumulationFrameBuffer;
};

Renderer::Renderer()
{
	m_CudaResourceAPI = new CudaAPI();
	m_CelestiumPTResourceAPI = new CelestiumPT_API();
	//TODO: redundant
	m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
	m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);

	cudaMallocManaged(&m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.DeviceGeometryAggregate, sizeof(SceneGeometry));

	m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.DeviceGeometryAggregate->SkyLight = InfiniteLight();

	m_CelestiumPTResourceAPI->DeviceScene = DeviceScene(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.DeviceGeometryAggregate);
	m_CurrentScene = HostScene(&(m_CelestiumPTResourceAPI->DeviceScene));
}

void Renderer::resizeResolution(int width, int height)
{
	if (width == m_NativeRenderResolutionWidth && height == m_NativeRenderResolutionHeight)return;
	m_NativeRenderResolutionHeight = height;
	m_NativeRenderResolutionWidth = width;

	m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->CompositeRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->AlbedoRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->PositionsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->DepthRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->GASDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->UVsDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->BarycentricsDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->ObjectIDRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->VelocityRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CelestiumPTResourceAPI->IrradianceRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->MomentsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	// History Buffers
	m_CelestiumPTResourceAPI->HistoryIntegratedMomentsFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryIntegratedMomentsBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
	m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);

	m_CelestiumPTResourceAPI->AccumulationFrameBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);
	m_CelestiumPTResourceAPI->VarianceAccumulationFrameBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);
}

static uint32_t g_frameIndex = 1;

void texCopy(const FrameBuffer& src, const FrameBuffer& dst, int t_width, int t_height) {
	glCopyImageSubData(
		src.m_RenderTargetTextureName, GL_TEXTURE_2D, 0, 0, 0, 0,
		dst.m_RenderTargetTextureName, GL_TEXTURE_2D, 0, 0, 0, 0,
		t_width, t_height, 1);
	glFinish();
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		printf("%s\n", glErrorString(err));
	}
}

void Renderer::renderFrame()
{
	//pre render-------------------------------------------------------------
	{
		m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_variance_render_front_surfobj));
		m_CelestiumPTResourceAPI->CompositeRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_render_surface_object));
		m_CelestiumPTResourceAPI->AlbedoRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.albedo_render_surface_object));
		m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.world_normals_render_surface_object));
		m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_normals_render_surface_object));
		m_CelestiumPTResourceAPI->PositionsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.positions_render_surface_object));
		m_CelestiumPTResourceAPI->DepthRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.depth_render_surface_object));
		m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_irradiance_front_surfobj));
		m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_irradiance_back_surfobj));
		m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_depth_render_surface_object));
		m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_world_normals_render_surface_object));
		m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_positions_render_surface_object));
		m_CelestiumPTResourceAPI->GASDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.GAS_debug_render_surface_object));
		m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.heatmap_debug_render_surface_object));
		m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.bbox_heatmap_debug_render_surface_object));
		m_CelestiumPTResourceAPI->UVsDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.UV_debug_render_surface_object));
		m_CelestiumPTResourceAPI->BarycentricsDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.bary_debug_render_surface_object));
		m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.objectID_debug_render_surface_object));
		m_CelestiumPTResourceAPI->ObjectIDRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.objectID_render_surface_object));
		m_CelestiumPTResourceAPI->VelocityRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.velocity_render_surface_object));
		m_CelestiumPTResourceAPI->IrradianceRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.current_irradiance_render_surface_object));
		m_CelestiumPTResourceAPI->MomentsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.current_moments_render_surface_object));
		m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_variance_render_back_surfobj));
		m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_irradiance_front_render_surface_object));
		m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_irradiance_back_render_surface_object));

		// History Buffers
		m_CelestiumPTResourceAPI->HistoryIntegratedMomentsFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_moments_front_surfobj));
		m_CelestiumPTResourceAPI->HistoryIntegratedMomentsBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_moments_back_surfobj));
	}

	//prepare globals--------------------
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameIndex = g_frameIndex;
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.resolution =
		make_int2(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.accumulation_framebuffer =
		thrust::raw_pointer_cast(m_CelestiumPTResourceAPI->AccumulationFrameBuffer.data());
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.variance_accumulation_framebuffer =
		thrust::raw_pointer_cast(m_CelestiumPTResourceAPI->VarianceAccumulationFrameBuffer.data());

	//Launch RenderChain

	if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.use_SVGF)
	{
		renderPathTraceRaw << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		//-----------
		temporalIntegrate << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		//moments
		texCopy(m_CelestiumPTResourceAPI->HistoryIntegratedMomentsBackBuffer,
			m_CelestiumPTResourceAPI->HistoryIntegratedMomentsFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//-----------

		for (int iter = 0; iter < 5; iter++) {
		
		}
		SVGFPass << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals, 1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		texCopy(m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer,
			m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		texCopy(m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer,
			m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		//irradiance feedback: TODO: basically bo need for hist_int_irr back buffer
		texCopy(m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer,
			m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		//-----------
		SVGFPass << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals, 2);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		texCopy(m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer,
			m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		texCopy(m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer,
			m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		//-----------
		SVGFPass << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals, 4);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		texCopy(m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer,
			m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		texCopy(m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer,
			m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		//-----------
		SVGFPass << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals, 8);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		texCopy(m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer,
			m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		texCopy(m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer,
			m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		//-----------
		composeCompositeImage << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);//Display!
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.temporal_accumulation)
	{
		renderPathTraceRaw << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		//---
		temporalAccumulate << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		texCopy(m_CelestiumPTResourceAPI->HistoryIntegratedMomentsBackBuffer,
			m_CelestiumPTResourceAPI->HistoryIntegratedMomentsFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		texCopy(m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderBackBuffer,
			m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		//---

		composeCompositeImage << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);//Display!
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		renderPathTraceRaw << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		texCopy(m_CelestiumPTResourceAPI->IrradianceRenderBuffer,
			m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
		composeCompositeImage << < m_CudaResourceAPI->m_BlockGridDimensions,
			m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);//Display!
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	g_frameIndex++;

	//----
	{
		//post render cuda---------------------------------------------------------------------------------
		m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_variance_render_front_surfobj));
		m_CelestiumPTResourceAPI->CompositeRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_render_surface_object));
		m_CelestiumPTResourceAPI->AlbedoRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.albedo_render_surface_object));
		m_CelestiumPTResourceAPI->UVsDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.UV_debug_render_surface_object));
		m_CelestiumPTResourceAPI->BarycentricsDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.bary_debug_render_surface_object));
		m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.world_normals_render_surface_object));
		m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_normals_render_surface_object));
		m_CelestiumPTResourceAPI->DepthRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.depth_render_surface_object));
		m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_irradiance_front_surfobj));
		m_CelestiumPTResourceAPI->HistoryIntegratedIrradianceRenderBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_irradiance_back_surfobj));
		m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_depth_render_surface_object));
		m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_world_normals_render_surface_object));
		m_CelestiumPTResourceAPI->PositionsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.positions_render_surface_object));
		m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_positions_render_surface_object));
		m_CelestiumPTResourceAPI->GASDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.GAS_debug_render_surface_object));
		m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.objectID_debug_render_surface_object));
		m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.bbox_heatmap_debug_render_surface_object));
		m_CelestiumPTResourceAPI->ObjectIDRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.objectID_render_surface_object));
		m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.heatmap_debug_render_surface_object));
		m_CelestiumPTResourceAPI->VelocityRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.velocity_render_surface_object));
		m_CelestiumPTResourceAPI->IrradianceRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.current_irradiance_render_surface_object));
		m_CelestiumPTResourceAPI->MomentsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.current_moments_render_surface_object));
		m_CelestiumPTResourceAPI->FilteredVarianceRenderBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_variance_render_back_surfobj));
		m_CelestiumPTResourceAPI->FilteredIrradianceFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_irradiance_front_render_surface_object));
		m_CelestiumPTResourceAPI->FilteredIrradianceBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.filtered_irradiance_back_render_surface_object));

		// History Buffers
		m_CelestiumPTResourceAPI->HistoryIntegratedMomentsFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_moments_front_surfobj));
		m_CelestiumPTResourceAPI->HistoryIntegratedMomentsBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_integrated_moments_back_surfobj));
	}

	//UPDATE HISTORY
	//GBUFFER blit

	//depth
	texCopy(m_CelestiumPTResourceAPI->DepthRenderBuffer,
		m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer, m_NativeRenderResolutionWidth,
		m_NativeRenderResolutionHeight);

	//normals
	texCopy(m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer,
		m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer, m_NativeRenderResolutionWidth,
		m_NativeRenderResolutionHeight);
}

void Renderer::clearAccumulation()
{
	if (!m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.accumulate)return;

	thrust::fill(m_CelestiumPTResourceAPI->AccumulationFrameBuffer.begin(),
		m_CelestiumPTResourceAPI->AccumulationFrameBuffer.end(), make_float3(0, 0, 0));
	thrust::fill(m_CelestiumPTResourceAPI->VarianceAccumulationFrameBuffer.begin(),
		m_CelestiumPTResourceAPI->VarianceAccumulationFrameBuffer.end(), make_float3(0, 0, 0));

	g_frameIndex = 1;
}

GLuint Renderer::getCompositeRenderTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->CompositeRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getAlbedoRenderTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->AlbedoRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getNormalsTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getPositionsTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->PositionsRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getLocalPositionsTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getVelocityTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->VelocityRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getGASDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->GASDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getHeatmapDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getBboxHeatmapDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getDepthTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->DepthRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getUVsDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->UVsDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getObjectIDDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getBarycentricsDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->BarycentricsDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getAlbedoTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->AlbedoRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getIntegratedVarianceTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->FilteredVarianceRenderFrontBuffer.m_RenderTargetTextureName;
}

int Renderer::getSPP() const
{
	return g_frameIndex;
}

void Renderer::setCamera(int idx)
{
	if (idx < 0)idx = 0;
	if (idx >= m_CelestiumPTResourceAPI->DeviceScene.DeviceCameras.size())idx = m_CelestiumPTResourceAPI->DeviceScene.DeviceCameras.size() - 1;
	DeviceCamera* dcam = thrust::raw_pointer_cast(&m_CelestiumPTResourceAPI->DeviceScene.DeviceCameras[idx]);
	m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.ActiveCamera = dcam;
	m_CurrentCamera = HostCamera(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.ActiveCamera);
}

IntegratorSettings* Renderer::getIntegratorSettings()
{
	return &(m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG);
}

Renderer::~Renderer()
{
	cudaFree(m_CelestiumPTResourceAPI->m_IntegratorGlobals.SceneDescriptor.DeviceGeometryAggregate);//TODO: non critical ownership issues with devicescene
	delete m_CudaResourceAPI;
	m_CelestiumPTResourceAPI->DeviceScene.DeviceCameras.clear();
	delete m_CelestiumPTResourceAPI;
}