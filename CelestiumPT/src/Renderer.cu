#include "renderer.hpp"
#include "error_check.cuh"
#include "storage.cuh"
#include "integrator.cuh"
#include "denoiser/denoiser.cuh"
#include "render_passes.cuh"
#include "maths/constants.cuh"

#include "scene_geometry.cuh"
#include "device_scene.cuh"
#include "device_camera.cuh"
//#include "Triangle.cuh"

#include <cuda_gl_interop.h>
#include <iostream>
/*
* Mostly adapter and inter conversion work
*/

constexpr int SVGF_MAX_ITERATIONS = 4;
constexpr int ASVGF_MAX_ITERATIONS = 5;

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

	//post
	FrameBuffer CompositeRenderBuffer;//srgb view transformed

	//raw
	FrameBuffer RawIrradianceRenderBuffer;

	//SVGF
	FrameBuffer SVGFFilteredIrradianceFrontBuffer;
	FrameBuffer SVGFFilteredIrradianceBackBuffer;
	FrameBuffer SVGFFilteredVarianceRenderFrontBuffer;
	FrameBuffer SVGFFilteredVarianceRenderBackBuffer;

	//GBUFFER
	FrameBuffer AlbedoRenderBuffer;
	FrameBuffer LocalNormalsRenderBuffer;//used for normals reject
	FrameBuffer LocalPositionsRenderBuffer;//used for reproj & depth reject
	FrameBuffer WorldNormalsRenderBuffer;
	FrameBuffer WorldPositionsRenderBuffer;

	FrameBuffer DepthRenderBuffer;//used for rejection

	FrameBuffer UVsRenderBuffer;
	FrameBuffer BarycentricsRenderBuffer;
	FrameBuffer ObjectIDRenderBuffer; //used for reproj & reject
	FrameBuffer TriangleIDRenderBuffer;
	FrameBuffer VelocityRenderBuffer;//used for reproj

	//adaptive filter
	FrameBuffer SeedsBuffer;

	FrameBuffer HistoryShadingBuffer;
	FrameBuffer HistorySeedsBuffer;

	FrameBuffer HistoryObjectIDBuffer;
	FrameBuffer HistoryTriangleIDBuffer;

	FrameBuffer HistoryLocalPositionsBuffer;
	FrameBuffer HistoryWorldPositionsBuffer;
	FrameBuffer HistoryLocalNormalsBuffer;

	FrameBuffer HistoryUVsBuffer;
	FrameBuffer HistoryBarycentricsBuffer;

	FrameBuffer SparseGradientBuffer;
	FrameBuffer DenseGradientFrontBuffer;
	FrameBuffer DenseGradientBackBuffer;

	//temporal filter
	FrameBuffer IntegratedMomentsFrontBuffer;
	FrameBuffer IntegratedMomentsBackBuffer;
	FrameBuffer IntegratedIrradianceRenderFrontBuffer;//read
	FrameBuffer IntegratedIrradianceRenderBackBuffer;//write; Filtered output

	//history
	FrameBuffer HistoryDepthRenderBuffer;
	FrameBuffer HistoryWorldNormalsRenderBuffer;

	//debug
	FrameBuffer MiscDebugViewBuffer;
	FrameBuffer ObjectIDDebugRenderBuffer;
	FrameBuffer GASDebugRenderBuffer;
	FrameBuffer HitHeatmapDebugRenderBuffer;
	FrameBuffer BboxHeatmapDebugRenderBuffer;

	//static accum
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

	//post
	m_CelestiumPTResourceAPI->CompositeRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//raw
	m_CelestiumPTResourceAPI->RawIrradianceRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//svgf
	m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->SVGFFilteredIrradianceFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->SVGFFilteredIrradianceBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//gbuffer
	m_CelestiumPTResourceAPI->AlbedoRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->WorldPositionsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CelestiumPTResourceAPI->DepthRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CelestiumPTResourceAPI->UVsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->BarycentricsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->ObjectIDRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->TriangleIDRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->VelocityRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//temporal filter
	m_CelestiumPTResourceAPI->IntegratedIrradianceRenderFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->IntegratedIrradianceRenderBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->IntegratedMomentsFrontBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->IntegratedMomentsBackBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//asvgf
	m_CelestiumPTResourceAPI->SparseGradientBuffer.resizeResolution(
		(m_NativeRenderResolutionWidth + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE,
		(m_NativeRenderResolutionHeight + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE);
	m_CelestiumPTResourceAPI->DenseGradientFrontBuffer.resizeResolution(
		(m_NativeRenderResolutionWidth + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE,
		(m_NativeRenderResolutionHeight + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE);
	m_CelestiumPTResourceAPI->DenseGradientBackBuffer.resizeResolution(
		(m_NativeRenderResolutionWidth + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE,
		(m_NativeRenderResolutionHeight + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE);

	m_CelestiumPTResourceAPI->SeedsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryShadingBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistorySeedsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	m_CelestiumPTResourceAPI->HistoryLocalPositionsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryWorldPositionsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryLocalNormalsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryUVsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryBarycentricsBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryTriangleIDBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryObjectIDBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//debugviews
	m_CelestiumPTResourceAPI->GASDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->MiscDebugViewBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	// History Buffers
	m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);
	m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer.resizeResolution(m_NativeRenderResolutionWidth, m_NativeRenderResolutionHeight);

	//static accum
	m_CelestiumPTResourceAPI->AccumulationFrameBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);
	m_CelestiumPTResourceAPI->VarianceAccumulationFrameBuffer.resize(m_NativeRenderResolutionHeight * m_NativeRenderResolutionWidth);

	//=============================
	m_CudaResourceAPI->m_BlockGridDimensions = dim3(m_NativeRenderResolutionWidth / m_ThreadBlock_x + 1, m_NativeRenderResolutionHeight / m_ThreadBlock_y + 1);
	m_CudaResourceAPI->m_ThreadBlockDimensions = dim3(m_ThreadBlock_x, m_ThreadBlock_y);
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
		printf("[TEXCOPY]: %s\n", glErrorString(err));
	}
}

void Renderer::renderFrame()
{
	//pre render-------------------------------------------------------------
	{
		// Post ---------------------------------
		m_CelestiumPTResourceAPI->CompositeRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_surfobject));

		// Raw Buffers ---------------------------------
		m_CelestiumPTResourceAPI->RawIrradianceRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.raw_irradiance_surfobject));

		// SVGF Buffers ---------------------------------
		m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_variance_front_surfobject));
		m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_variance_back_surfobject));
		m_CelestiumPTResourceAPI->SVGFFilteredIrradianceFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_irradiance_front_surfobject));
		m_CelestiumPTResourceAPI->SVGFFilteredIrradianceBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_irradiance_back_surfobject));

		// Albedo Buffer ---------------------------------
		m_CelestiumPTResourceAPI->AlbedoRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.albedo_surfobject));

		// G-Buffer ---------------------------------
		m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_normals_surfobject));
		m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_positions_surfobject));
		m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.world_normals_surfobject));
		m_CelestiumPTResourceAPI->WorldPositionsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.world_positions_surfobject));
		m_CelestiumPTResourceAPI->DepthRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.depth_surfobject));
		m_CelestiumPTResourceAPI->UVsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.UVs_surfobject));
		m_CelestiumPTResourceAPI->BarycentricsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.bary_surfobject));
		m_CelestiumPTResourceAPI->ObjectIDRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.objectID_surfobject));
		m_CelestiumPTResourceAPI->TriangleIDRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.triangleID_surfobject));
		m_CelestiumPTResourceAPI->VelocityRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.velocity_surfobject));

		// Temporal Accumulation Buffers ---------------------------------
		m_CelestiumPTResourceAPI->IntegratedIrradianceRenderFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_irradiance_front_surfobject));
		m_CelestiumPTResourceAPI->IntegratedIrradianceRenderBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_irradiance_back_surfobject));
		m_CelestiumPTResourceAPI->IntegratedMomentsFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_moments_front_surfobject));
		m_CelestiumPTResourceAPI->IntegratedMomentsBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_moments_back_surfobject));

		// ASVGF -------------------------------
		m_CelestiumPTResourceAPI->SparseGradientBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.asvgf_sparse_gradient_surfobject));
		m_CelestiumPTResourceAPI->DenseGradientFrontBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.asvgf_dense_gradient_front_surfobject));
		m_CelestiumPTResourceAPI->DenseGradientBackBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.asvgf_dense_gradient_back_surfobject));

		m_CelestiumPTResourceAPI->SeedsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.seeds_surfobject));
		m_CelestiumPTResourceAPI->HistoryShadingBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_shading_surfobject));
		m_CelestiumPTResourceAPI->HistorySeedsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_seeds_surfobject));

		m_CelestiumPTResourceAPI->HistoryLocalPositionsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_local_positions_surfobject));
		m_CelestiumPTResourceAPI->HistoryWorldPositionsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_world_positions_surfobject));
		m_CelestiumPTResourceAPI->HistoryLocalNormalsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_local_normals_surfobject));
		m_CelestiumPTResourceAPI->HistoryUVsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_UVs_surfobject));
		m_CelestiumPTResourceAPI->HistoryBarycentricsBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_bary_surfobject));
		m_CelestiumPTResourceAPI->HistoryTriangleIDBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_triangleID_surfobject));
		m_CelestiumPTResourceAPI->HistoryObjectIDBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_objectID_surfobject));

		// History Buffers ---------------------------------
		m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_depth_surfobject));
		m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_world_normals_surfobject));

		// Debug Buffers ---------------------------------
		m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_objectID_surfobject));
		m_CelestiumPTResourceAPI->MiscDebugViewBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_misc_surfobject));

		m_CelestiumPTResourceAPI->GASDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_GAS_overlap_surfobject));
		m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_tri_test_heatmap_surfobject));
		m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.beginRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_bbox_test_heatmap_surfobject));
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
	{
		//Compute primary visbility=========================================
		{
			computePrimaryVisibility << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		//Merge shading and surface samples==========================================
		if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.adaptive_temporal_filter_enabled)
		{
			mergeSamples << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		//Compute pathtrace samples==========================================
		{
			tracePathSample << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			//potential skips----------------
			if (!m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.svgf_enabled &&
				!m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.temporal_filter_enabled)
				texCopy(m_CelestiumPTResourceAPI->RawIrradianceRenderBuffer,
					m_CelestiumPTResourceAPI->SVGFFilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
					m_NativeRenderResolutionHeight);
		}

		//Compute gradient samples==========================================
		if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.adaptive_temporal_filter_enabled)
		{
			createGradientSamples << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			texCopy(m_CelestiumPTResourceAPI->SparseGradientBuffer,
				m_CelestiumPTResourceAPI->DenseGradientFrontBuffer,
				(m_NativeRenderResolutionWidth + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE,
				(m_NativeRenderResolutionHeight + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE);
		}

		//Gradient atrous===================================================
		if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.adaptive_temporal_filter_enabled)
		{
			for (int iter = 0; iter < ASVGF_MAX_ITERATIONS; iter++)
			{
				int stepsize = pow(2, iter);

				atrousGradient << < m_CudaResourceAPI->m_BlockGridDimensions,
					m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals, stepsize);
				//sync-------
				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());

				//working buffers swap----------
				texCopy(m_CelestiumPTResourceAPI->DenseGradientBackBuffer,
					m_CelestiumPTResourceAPI->DenseGradientFrontBuffer,
					(m_NativeRenderResolutionWidth + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE,
					(m_NativeRenderResolutionHeight + ASVGF_STRATUM_SIZE - 1) / ASVGF_STRATUM_SIZE);
			}
		}

		//Temporal accumulation===========================================
		if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.svgf_enabled ||
			m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.temporal_filter_enabled)
		{
			temporalAccumulate << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			//moments swap
			texCopy(m_CelestiumPTResourceAPI->IntegratedMomentsBackBuffer,
				m_CelestiumPTResourceAPI->IntegratedMomentsFrontBuffer, m_NativeRenderResolutionWidth,
				m_NativeRenderResolutionHeight);
			//potential irradiance feedback
			if (!m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.svgf_enabled) {
				texCopy(m_CelestiumPTResourceAPI->SVGFFilteredIrradianceFrontBuffer,
					m_CelestiumPTResourceAPI->IntegratedIrradianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
					m_NativeRenderResolutionHeight);
			}

			//Variance Estimation=================================================
			estimateVariance << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}

		//DEBUG
		//texCopy(m_CelestiumPTResourceAPI->RawIrradianceRenderBuffer,
		//	m_CelestiumPTResourceAPI->MiscDebugViewBuffer, m_NativeRenderResolutionWidth,
		//	m_NativeRenderResolutionHeight);

		//SVGF atrous====================================
		if (m_CelestiumPTResourceAPI->m_IntegratorGlobals.IntegratorCFG.svgf_enabled)
		{
			for (int iter = 0; iter < SVGF_MAX_ITERATIONS; iter++)
			{
				int stepsize = pow(2, iter);

				atrousSVGF << < m_CudaResourceAPI->m_BlockGridDimensions,
					m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals, stepsize);
				//sync-------
				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());

				//irradiance feedback after 1st iter: TODO: basically no need for hist_int_irr back buffer
				if (iter == 0) {
					texCopy(m_CelestiumPTResourceAPI->SVGFFilteredIrradianceBackBuffer,
						m_CelestiumPTResourceAPI->IntegratedIrradianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
						m_NativeRenderResolutionHeight);
				}

				//working buffers swap----------
				texCopy(m_CelestiumPTResourceAPI->SVGFFilteredIrradianceBackBuffer,
					m_CelestiumPTResourceAPI->SVGFFilteredIrradianceFrontBuffer, m_NativeRenderResolutionWidth,
					m_NativeRenderResolutionHeight);
				texCopy(m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderBackBuffer,
					m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderFrontBuffer, m_NativeRenderResolutionWidth,
					m_NativeRenderResolutionHeight);
			}
		}

		//Compose=====================
		{
			composeCompositeImage << < m_CudaResourceAPI->m_BlockGridDimensions,
				m_CudaResourceAPI->m_ThreadBlockDimensions >> > (m_CelestiumPTResourceAPI->m_IntegratorGlobals);
			//sync
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}

	g_frameIndex++;

	//post render cuda---------------------------------------------------------------------------------
	{
		// Post ---------------------------------
		m_CelestiumPTResourceAPI->CompositeRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.composite_surfobject));

		// Raw Buffers ---------------------------------
		m_CelestiumPTResourceAPI->RawIrradianceRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.raw_irradiance_surfobject));

		// SVGF Buffers ---------------------------------
		m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_variance_front_surfobject));
		m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_variance_back_surfobject));
		m_CelestiumPTResourceAPI->SVGFFilteredIrradianceFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_irradiance_front_surfobject));
		m_CelestiumPTResourceAPI->SVGFFilteredIrradianceBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.svgf_filtered_irradiance_back_surfobject));

		// Albedo Buffer ---------------------------------
		m_CelestiumPTResourceAPI->AlbedoRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.albedo_surfobject));

		// G-Buffer ---------------------------------
		m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_normals_surfobject));
		m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.local_positions_surfobject));
		m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.world_normals_surfobject));
		m_CelestiumPTResourceAPI->WorldPositionsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.world_positions_surfobject));
		m_CelestiumPTResourceAPI->DepthRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.depth_surfobject));
		m_CelestiumPTResourceAPI->UVsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.UVs_surfobject));
		m_CelestiumPTResourceAPI->BarycentricsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.bary_surfobject));
		m_CelestiumPTResourceAPI->ObjectIDRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.objectID_surfobject));
		m_CelestiumPTResourceAPI->TriangleIDRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.triangleID_surfobject));
		m_CelestiumPTResourceAPI->VelocityRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.velocity_surfobject));

		// Temporal Accumulation Buffers ---------------------------------
		m_CelestiumPTResourceAPI->IntegratedIrradianceRenderFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_irradiance_front_surfobject));
		m_CelestiumPTResourceAPI->IntegratedIrradianceRenderBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_irradiance_back_surfobject));
		m_CelestiumPTResourceAPI->IntegratedMomentsFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_moments_front_surfobject));
		m_CelestiumPTResourceAPI->IntegratedMomentsBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.integrated_moments_back_surfobject));

		//ASVGF---------------------------------------------
		m_CelestiumPTResourceAPI->SparseGradientBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.asvgf_sparse_gradient_surfobject));
		m_CelestiumPTResourceAPI->DenseGradientFrontBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.asvgf_dense_gradient_front_surfobject));
		m_CelestiumPTResourceAPI->DenseGradientBackBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.asvgf_dense_gradient_back_surfobject));

		m_CelestiumPTResourceAPI->SeedsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.seeds_surfobject));
		m_CelestiumPTResourceAPI->HistoryShadingBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_shading_surfobject));
		m_CelestiumPTResourceAPI->HistorySeedsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_seeds_surfobject));

		m_CelestiumPTResourceAPI->HistoryLocalPositionsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_local_positions_surfobject));
		m_CelestiumPTResourceAPI->HistoryWorldPositionsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_world_positions_surfobject));
		m_CelestiumPTResourceAPI->HistoryLocalNormalsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_local_normals_surfobject));
		m_CelestiumPTResourceAPI->HistoryUVsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_UVs_surfobject));
		m_CelestiumPTResourceAPI->HistoryBarycentricsBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_bary_surfobject));
		m_CelestiumPTResourceAPI->HistoryTriangleIDBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_triangleID_surfobject));
		m_CelestiumPTResourceAPI->HistoryObjectIDBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_objectID_surfobject));

		// History Buffers ---------------------------------
		m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_depth_surfobject));
		m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.history_world_normals_surfobject));

		// Debug Buffers ---------------------------------
		m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_objectID_surfobject));
		m_CelestiumPTResourceAPI->MiscDebugViewBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_misc_surfobject));
		m_CelestiumPTResourceAPI->GASDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_GAS_overlap_surfobject));
		m_CelestiumPTResourceAPI->HitHeatmapDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_tri_test_heatmap_surfobject));
		m_CelestiumPTResourceAPI->BboxHeatmapDebugRenderBuffer.endRender(
			&(m_CelestiumPTResourceAPI->m_IntegratorGlobals.FrameBuffer.debugview_bbox_test_heatmap_surfobject));
	}

	//UPDATE HISTORY
	{
		//seeds
		texCopy(m_CelestiumPTResourceAPI->SeedsBuffer,
			m_CelestiumPTResourceAPI->HistorySeedsBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//shading
		texCopy(m_CelestiumPTResourceAPI->RawIrradianceRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryShadingBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//w_normals
		texCopy(m_CelestiumPTResourceAPI->WorldNormalsRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryWorldNormalsRenderBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//w_pos
		texCopy(m_CelestiumPTResourceAPI->WorldPositionsRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryWorldPositionsBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//l_nrm
		texCopy(m_CelestiumPTResourceAPI->LocalNormalsRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryLocalNormalsBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//l_pos
		texCopy(m_CelestiumPTResourceAPI->LocalPositionsRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryLocalPositionsBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//depth
		texCopy(m_CelestiumPTResourceAPI->DepthRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryDepthRenderBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//triangle ID
		texCopy(m_CelestiumPTResourceAPI->TriangleIDRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryTriangleIDBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//object ID
		texCopy(m_CelestiumPTResourceAPI->ObjectIDRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryObjectIDBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//uvs
		texCopy(m_CelestiumPTResourceAPI->UVsRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryUVsBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);

		//bary
		texCopy(m_CelestiumPTResourceAPI->BarycentricsRenderBuffer,
			m_CelestiumPTResourceAPI->HistoryBarycentricsBuffer, m_NativeRenderResolutionWidth,
			m_NativeRenderResolutionHeight);
	}
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
	return m_CelestiumPTResourceAPI->WorldPositionsRenderBuffer.m_RenderTargetTextureName;
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
	return m_CelestiumPTResourceAPI->UVsRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getObjectIDDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->ObjectIDDebugRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getBarycentricsDebugTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->BarycentricsRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getAlbedoTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->AlbedoRenderBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getIntegratedVarianceTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->SVGFFilteredVarianceRenderFrontBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getSparseGradientTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->SparseGradientBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getDenseGradientTargetTextureName() const
{
	return m_CelestiumPTResourceAPI->DenseGradientFrontBuffer.m_RenderTargetTextureName;
}

GLuint Renderer::getMiscDebugTextureName() const
{
	return m_CelestiumPTResourceAPI->MiscDebugViewBuffer.m_RenderTargetTextureName;
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