#pragma once

#include "ErrorCheck.cuh"

#include <glad/glad.h>
#define __CUDACC__
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include <vector_types.h>

#include <type_traits>

struct TextureBuffer {
	__host__ TextureBuffer() = default;

	__host__ TextureBuffer(cudaSurfaceObject_t t_cuda_surf, int t_width, int t_height, cudaSurfaceObject_t t_front_surf) :
		m_cuda_surface(t_cuda_surf), m_width(t_width), m_height(t_height), m_cuda_front_surface(t_front_surf) {};

	__device__ float4 texRead(int2 pix) const {
		float4 data;
		if (m_cuda_front_surface != NULL)
			data = surf2Dread<float4>(m_cuda_front_surface,
				pix.x * (int)sizeof(float4), pix.y);
		else
			data = surf2Dread<float4>(m_cuda_surface,
				pix.x * (int)sizeof(float4), pix.y);

		return data;
	}

	//has to be uchar4/2/1 or float4/2/1; no 3 comp color
	__device__ void texWrite(float4 data, int2 pix) {
		surf2Dwrite<float4>(data, m_cuda_surface, pix.x * (int)sizeof(float4), pix.y);
	}

	//private:
	cudaSurfaceObject_t m_cuda_surface = NULL;
	cudaSurfaceObject_t m_cuda_front_surface = NULL;
	int m_width = 0, m_height = 0;
};

class Buffer {
public:

	virtual void init(int t_width = 16, int t_height = 16) = 0;
	virtual void destroy() = 0;
	virtual void resize(int t_width, int t_height) = 0;
	virtual void allowCudaAccess() = 0;
	virtual void shutOffCudaAccess() = 0;
	virtual TextureBuffer getTextureBuffer() const = 0;

public:
	cudaSurfaceObject_t m_cuda_surface;
	cudaGraphicsResource_t m_cuda_graphics_resource;
	GLuint m_gl_tex_name = 0;

	int m_width = 0, m_height = 0;
};

//internal formats=uchar/float/1/2/4
class GenericBuffer : public Buffer
{
public:
	void init(int t_width = 16, int t_height = 16) override {
		if (m_gl_tex_name != 0)return;
		m_width = t_width; m_height = t_height;

		//GL texture configure
		glGenTextures(1, &m_gl_tex_name);
		glBindTexture(GL_TEXTURE_2D, m_gl_tex_name);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		//TODO: make a switchable frame filtering mode
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, t_width, t_height, 0, GL_RGBA, GL_FLOAT, NULL);

		glBindTexture(GL_TEXTURE_2D, 0);

		checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cuda_graphics_resource,
			m_gl_tex_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	};

	//called only after shutOffCudaAccess()
	void resize(int t_width, int t_height) override {
		if (m_cuda_access_enabled) {
			shutOffCudaAccess();
		}

		if (t_width == m_width && t_height == m_height) {
			return;
		}
		if (m_gl_tex_name == 0) {
			init(t_width, t_height); return;
		}
		checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_graphics_resource));//not needed here but okay
		// resize
		glBindTexture(GL_TEXTURE_2D, m_gl_tex_name);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, t_width, t_height, 0, GL_RGBA, GL_FLOAT, NULL);
		}
		glBindTexture(GL_TEXTURE_2D, 0);

		checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cuda_graphics_resource,
			m_gl_tex_name, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void copy(GenericBuffer t_buff) {
		if (t_buff.m_gl_tex_name == 0)return;
		glBindTexture(GL_TEXTURE_2D, t_buff.m_gl_tex_name);
		glCopyImageSubData(
			t_buff.m_gl_tex_name, GL_TEXTURE_2D, 0, 0, 0, 0,
			m_gl_tex_name, GL_TEXTURE_2D, 0, 0, 0, 0,
			t_buff.m_width, t_buff.m_height, 0);
		glFinish();
		glBindTexture(GL_TEXTURE_2D, 0);
		GLenum err = glGetError();
		if (err != GL_NO_ERROR) {
			glErrorString(err);
		}
	}

	//call shutOffCudaAccess() after calling this
	void allowCudaAccess() override {
		checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_graphics_resource));

		cudaArray_t render_target_texture_sub_resource_array;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&render_target_texture_sub_resource_array, m_cuda_graphics_resource,
			0, 0));
		cudaResourceDesc render_target_texture_resource_descriptor;
		{
			render_target_texture_resource_descriptor.resType = cudaResourceTypeArray;
			render_target_texture_resource_descriptor.res.array.array = render_target_texture_sub_resource_array;
		}

		checkCudaErrors(cudaCreateSurfaceObject(&m_cuda_surface, &render_target_texture_resource_descriptor));
		m_cuda_access_enabled = true;
	}

	//always called after allowCudaAccess() is called
	void shutOffCudaAccess() override {
		checkCudaErrors(cudaDestroySurfaceObject(m_cuda_surface));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_graphics_resource));
		//checkCudaErrors(cudaStreamSynchronize(0));
		m_cuda_access_enabled = false;
	}

	void destroy() override {
		if (m_gl_tex_name != 0) {
			if (m_cuda_access_enabled) shutOffCudaAccess();
			checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_graphics_resource));
			glDeleteTextures(1, &m_gl_tex_name);
			m_gl_tex_name = 0;
		}
	};

	TextureBuffer getTextureBuffer() const override {
		return TextureBuffer(m_cuda_surface, m_width, m_height, NULL);
	}

	GLuint getGLName() const { return m_gl_tex_name; }

	~GenericBuffer() {
		destroy();
	}

public:

	bool m_cuda_access_enabled = false;
};

class RWBuffer : public Buffer {
public:

	void init(int t_width = 16, int t_height = 16) override {
		front_buffer.init(t_width, t_height);
		back_buffer.init(t_width, t_height);
	}

	void resize(int t_width, int t_height)override {
		front_buffer.resize(t_width, t_height);
		back_buffer.resize(t_width, t_height);
	}

	void allowCudaAccess()override {
		front_buffer.allowCudaAccess();
		back_buffer.allowCudaAccess();
	};

	void shutOffCudaAccess()override {
		front_buffer.shutOffCudaAccess();
		back_buffer.shutOffCudaAccess();
	};

	void destroy()override
	{
		front_buffer.destroy();
		back_buffer.destroy();
	};

	TextureBuffer getTextureBuffer() const override {
		return TextureBuffer(back_buffer.m_cuda_surface, back_buffer.m_width,
			back_buffer.m_height, front_buffer.m_cuda_surface);
	}

	void apply() {
		cudaDeviceSynchronize();
		front_buffer.copy(back_buffer);
		cudaDeviceSynchronize();
	};

	GLuint getBackGLName() const { return back_buffer.getGLName(); }
	GLuint getFrontGLName() const { return front_buffer.getGLName(); }

	GenericBuffer front_buffer;
	GenericBuffer back_buffer;
};

class GBuffer {
public:
	void init(int t_width = 16, int t_height = 16) {
		albedo.init(t_width, t_height);
		local_normals.init(t_width, t_height);
		world_normals.init(t_width, t_height);
		depth.init(t_width, t_height);
		objectID.init(t_width, t_height);
		local_position.init(t_width, t_height);
		velocity.init(t_width, t_height);
	}

	void resize(int t_width, int t_height) {
		albedo.resize(t_width, t_height);
		local_normals.resize(t_width, t_height);
		world_normals.resize(t_width, t_height);
		depth.resize(t_width, t_height);
		objectID.resize(t_width, t_height);
		local_position.resize(t_width, t_height);
		velocity.resize(t_width, t_height);
	}

	void destroy() {
		if (albedo.m_gl_tex_name != 0) albedo.destroy();
		if (local_normals.m_gl_tex_name != 0) local_normals.destroy();
		if (world_normals.m_gl_tex_name != 0) world_normals.destroy();
		if (depth.m_gl_tex_name != 0) depth.destroy();
		if (objectID.m_gl_tex_name != 0) objectID.destroy();
		if (local_position.m_gl_tex_name != 0) local_position.destroy();
		if (velocity.m_gl_tex_name != 0) velocity.destroy();
	}


	GenericBuffer albedo;
	GenericBuffer local_normals;
	GenericBuffer world_normals;
	GenericBuffer depth;
	GenericBuffer objectID;
	GenericBuffer local_position;
	GenericBuffer velocity;
};

class HistoryBuffer {
public:
	void init(int t_width = 16, int t_height = 16) {
		world_normals.init(t_width, t_height);
		depth.init(t_width, t_height);
		//objectID.init(t_width, t_height);
	}

	void resize(int t_width, int t_height) {
		world_normals.resize(t_width, t_height);
		depth.resize(t_width, t_height);
		//objectID.resize(t_width, t_height);
	}

	void destroy() {
		world_normals.destroy();
		depth.destroy();
		//objectID.destroy();
	}

	void update(const GBuffer* t_current_buffer) {
		world_normals.copy(t_current_buffer->world_normals);
		depth.copy(t_current_buffer->depth);
		//objectID.copy(t_current_buffer->objectID);
	};

	GenericBuffer world_normals;
	GenericBuffer depth;
	//GenericBuffer objectID;
};

struct DeviceTextureBuffers {
	struct HistoryTextureBuffers {
		HistoryTextureBuffers() : depth(TextureBuffer()), world_normals(TextureBuffer()) {}
		TextureBuffer depth;
		TextureBuffer world_normals;
	};

	TextureBuffer composite;
	TextureBuffer raw_irradiance;
	TextureBuffer raw_moments;

	HistoryTextureBuffers history_data;

	TextureBuffer albedo;
	TextureBuffer local_normals;
	TextureBuffer world_normals;
	TextureBuffer depth;
	TextureBuffer objectID;
	TextureBuffer local_position;
	TextureBuffer velocity;

	TextureBuffer filtered_irradiance;
	TextureBuffer filtered_variance;

	TextureBuffer integrated_irradiance;
	TextureBuffer integrated_moments;
};

class FrameBuffer {
public:

	//call after any allowCudaAccess() calls; DTBs are for sending off to device;
	DeviceTextureBuffers updateTextureBuffers() {
		DeviceTextureBuffers dtb;
		dtb.composite = Composite.getTextureBuffer();
		dtb.raw_irradiance = RawIrradiance.getTextureBuffer();
		dtb.raw_moments = RawMoments.getTextureBuffer();
		dtb.albedo = Gbuffer.albedo.getTextureBuffer();
		dtb.local_normals = Gbuffer.local_normals.getTextureBuffer();
		dtb.world_normals = Gbuffer.world_normals.getTextureBuffer();
		dtb.depth = Gbuffer.depth.getTextureBuffer();
		dtb.objectID = Gbuffer.objectID.getTextureBuffer();
		dtb.local_position = Gbuffer.local_position.getTextureBuffer();
		dtb.velocity = Gbuffer.velocity.getTextureBuffer();
		dtb.filtered_irradiance = FilteredIrradiance.getTextureBuffer();
		dtb.filtered_variance = FilteredVariance.getTextureBuffer();
		dtb.integrated_irradiance = IntegratedIrradiance.getTextureBuffer();
		dtb.integrated_moments = IntegratedMoments.getTextureBuffer();

		DeviceTextureBuffers::HistoryTextureBuffers htb;
		htb.depth = history_buffer.depth.getTextureBuffer();
		htb.world_normals = history_buffer.world_normals.getTextureBuffer();
		dtb.history_data = htb;

		return dtb;
	}

	GenericBuffer Composite;//Final display
	GenericBuffer RawIrradiance;
	GenericBuffer RawMoments;

	HistoryBuffer history_buffer;

	GBuffer Gbuffer;
	//svgf
	RWBuffer FilteredIrradiance;
	RWBuffer FilteredVariance;

	//TAA
	RWBuffer IntegratedIrradiance;
	RWBuffer IntegratedMoments;
};