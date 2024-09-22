#pragma once

#include "IntegratorSettings.hpp"
#include "HostCamera.hpp"
#include "HostScene.hpp"

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
	GLuint getNormalsTargetTextureName() const;
	GLuint getPositionsTargetTextureName() const;
	GLuint getGASDebugTargetTextureName() const;
	GLuint getUVsDebugTargetTextureName() const;
	GLuint getBarycentricsDebugTargetTextureName() const;

	void setCamera(int idx);
	IntegratorSettings* getIntegratorSettings();//TODO: make this safer and more robust
	HostCamera* getCurrentCamera() { return &m_CurrentCamera; };
	HostScene* getCurrentScene() { return &m_CurrentScene; };

	uint32_t getFrameWidth() const { return m_NativeRenderResolutionWidth; }
	uint32_t getFrameHeight() const { return m_NativeRenderResolutionHeight; }

	~Renderer();

private:

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