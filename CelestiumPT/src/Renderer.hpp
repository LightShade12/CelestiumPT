#pragma once

//#include <Windows.h>
#include <glad/glad.h>
#include <cstdint>

struct CudaAPI;

class Renderer
{
public:
	Renderer();

	void resizeResolution(int width, int height);

	void renderFrame();

	GLuint getCompositeRenderTargetTextureName() const { return m_CompositeRenderTargetTextureName; }

	uint32_t getFrameWidth() const { return m_NativeRenderResolutionWidth; }
	uint32_t getFrameHeight() const { return m_NativeRenderResolutionHeight; }

	~Renderer();

private:
	GLuint m_CompositeRenderTargetTextureName = NULL;
	uint32_t m_NativeRenderResolutionWidth = NULL;
	uint32_t m_NativeRenderResolutionHeight = NULL;

	CudaAPI* m_CudaResourceAPI = nullptr;

	//cudaEvent_t start, stop;

	int m_ThreadBlock_x = 8;
	int m_ThreadBlock_y = 8;
};