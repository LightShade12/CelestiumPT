#pragma once

#include "integrator_settings.hpp"
#include "host_camera.hpp"
#include "host_scene.hpp"

#include <glad/glad.h>
#include <cstdint>

struct CudaAPI;
struct CelestiumPT_API;

class Renderer
{
public:
	Renderer();

	void resizeResolution(int width, int height);

	void renderFrame();

	void clearAccumulation();

	GLuint getCompositeRenderTargetTextureName() const;
	GLuint getAlbedoRenderTargetTextureName() const;
	GLuint getNormalsTargetTextureName() const;
	GLuint getPositionsTargetTextureName() const;
	GLuint getLocalPositionsTargetTextureName() const;
	GLuint getVelocityTargetTextureName() const;
	GLuint getGASDebugTargetTextureName() const;
	GLuint getHeatmapDebugTargetTextureName() const;
	GLuint getBboxHeatmapDebugTargetTextureName() const;
	GLuint getDepthTargetTextureName() const;
	GLuint getUVsDebugTargetTextureName() const;
	GLuint getObjectIDDebugTargetTextureName() const;
	GLuint getBarycentricsDebugTargetTextureName() const;
	GLuint getAlbedoTargetTextureName() const;
	GLuint getIntegratedVarianceTargetTextureName() const;

	int getSPP() const;
	void setCamera(int idx);
	IntegratorSettings* getIntegratorSettings();//TODO: make this safer and more robust
	HostCamera* getCurrentCamera() { return &m_CurrentCamera; };
	HostScene* getCurrentScene() { return &m_CurrentScene; };

	uint32_t getFrameWidth() const { return m_NativeRenderResolutionWidth; }
	uint32_t getFrameHeight() const { return m_NativeRenderResolutionHeight; }

	~Renderer();

private:

	void blitFilteredIrradianceVarianceBackToFront();

	void blitMomentsBackToFront();

	void blitFilteredIrradianceToHistory(bool read_from_back);

	GLuint m_blit_mediator_FBO0_name = NULL;
	GLuint m_blit_mediator_FBO1_name = NULL;

	//TODO:rename
	GLenum m_blit_target0_attachment = GL_COLOR_ATTACHMENT1;
	GLenum m_blit_target1_attachment = GL_COLOR_ATTACHMENT3;
	GLenum m_blit_target2_attachment = GL_COLOR_ATTACHMENT5;
	GLenum m_blit_target3_attachment = GL_COLOR_ATTACHMENT0;

	uint32_t m_NativeRenderResolutionWidth = NULL;
	uint32_t m_NativeRenderResolutionHeight = NULL;
	HostCamera m_CurrentCamera;
	HostScene m_CurrentScene;
	CudaAPI* m_CudaResourceAPI = nullptr;
	CelestiumPT_API* m_CelestiumPTResourceAPI = nullptr;

	//cudaEvent_t start, stop;

	int m_ThreadBlock_x = 8;
	int m_ThreadBlock_y = 8;
};