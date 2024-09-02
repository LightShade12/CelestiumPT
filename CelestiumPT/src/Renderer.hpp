#pragma once

//#include <Windows.h>
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

	GLuint getCompositeRenderTargetTextureName() const { return m_CompositeRenderTargetTextureName; }
	
	HostCamera* getCurrentCamera() { return &m_CurrentCamera; };
	HostScene* getCurrentScene() { return &m_CurrentScene; };

	uint32_t getFrameWidth() const { return m_NativeRenderResolutionWidth; }
	uint32_t getFrameHeight() const { return m_NativeRenderResolutionHeight; }

	~Renderer();

private:
	GLuint m_CompositeRenderTargetTextureName = NULL;
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