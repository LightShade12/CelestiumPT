#include "Integrator.cuh"
#include "SceneGeometry.cuh"
#include "DeviceCamera.cuh"
#include "Storage.cuh"
#include "RayStages.cuh"
//#include "Triangle.cuh"
#include "Ray.cuh"
//#include "DeviceMesh.cuh"
#include "ShapeIntersection.cuh"
#include "BSDF.cuh"
#include "acceleration_structure/GAS.cuh"
#include "Samplers.cuh"

#include "maths/maths_linear_algebra.cuh"
#include "maths/constants.cuh"

#include <device_launch_parameters.h>
#define __CUDACC__
#include <surface_indirect_functions.h>
#include <float.h>

void IntegratorPipeline::invokeRenderKernel(const IntegratorGlobals& globals, dim3 block_grid_dims, dim3 thread_block_dims)
{
	renderKernel << < block_grid_dims, thread_block_dims >> > (globals);
};

__global__ void renderKernel(IntegratorGlobals globals)
{
	int thread_pixel_coord_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_pixel_coord_y = threadIdx.y + blockIdx.y * blockDim.y;
	int2 frameres = globals.FrameBuffer.resolution;

	if ((thread_pixel_coord_x >= frameres.x) || (thread_pixel_coord_y >= frameres.y)) return;

	float3 sampled_radiance = IntegratorPipeline::evaluatePixelSample(globals, { (float)thread_pixel_coord_x,(float)thread_pixel_coord_y });

	if (globals.IntegratorCFG.accumulate) {
		globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] += sampled_radiance;
		sampled_radiance = globals.FrameBuffer.accumulation_framebuffer
			[thread_pixel_coord_x + thread_pixel_coord_y * globals.FrameBuffer.resolution.x] / (globals.frameidx);
	}

	float4 fragcolor = { sampled_radiance.x,sampled_radiance.y,sampled_radiance.z, 1 };

	//EOTF
	fragcolor = make_float4(sqrtf(sampled_radiance.x), sqrtf(sampled_radiance.y), sqrtf(sampled_radiance.z), 1);

	surf2Dwrite(fragcolor, globals.FrameBuffer.composite_render_surface_object, thread_pixel_coord_x * (int)sizeof(float4), thread_pixel_coord_y);//has to be uchar4/2/1 or float4/2/1; no 3 comp color
}

__device__ float3 IntegratorPipeline::evaluatePixelSample(const IntegratorGlobals& globals, float2 ppixel)
{
	uint32_t seed = ppixel.x + ppixel.y * globals.FrameBuffer.resolution.x;
	seed *= globals.frameidx;

	int2 frameres = globals.FrameBuffer.resolution;

	float2 screen_uv = { (ppixel.x / frameres.x),(ppixel.y / frameres.y) };
	screen_uv = screen_uv * 2 - 1;//-1->1

	Ray primary_ray = globals.SceneDescriptor.dev_camera->generateRay(frameres.x, frameres.y, screen_uv);

	float3 L = IntegratorPipeline::Li(globals, primary_ray, seed, ppixel);

	return L;
}

__device__ ShapeIntersection IntegratorPipeline::Intersect(const IntegratorGlobals& globals, const Ray& ray)
{
	ShapeIntersection payload;
	payload.hit_distance = FLT_MAX;
	Mat4 closest_hit_model_transform(1);
	Ray valid_transformed_ray = ray;

	/*for (int blasidx = 0; blasidx < globals.SceneDescriptor.dev_aggregate->DeviceBLASesCount; blasidx++) {
		BLAS* blas = &(globals.SceneDescriptor.dev_aggregate->DeviceBLASesBuffer[blasidx]);
		Mat4 modelmat = blas->m_MeshLink->modelMatrix;
		Ray transformedRay = ray;
		transformedRay.setOrigin(modelmat * make_float4(transformedRay.getOrigin(), 1));
		transformedRay.setDirection(normalize(modelmat * make_float4(transformedRay.getDirection(), 0)));
		ShapeIntersection eval_payload;
		eval_payload.hit_distance = FLT_MAX;

		blas->intersect(globals, ray, &eval_payload);

		if (eval_payload.hit_distance < payload.hit_distance && eval_payload.triangle_idx != -1) {
			payload = eval_payload;
			closest_hit_model_transform = modelmat;
			valid_transformed_ray = transformedRay;
		}
	};*/

	payload = globals.SceneDescriptor.dev_aggregate->GAS_structure.intersect(globals, ray);

	if (payload.triangle_idx == -1) {
		return MissStage(globals, ray, payload);
	}

	return ClosestHitStage(globals, valid_transformed_ray, closest_hit_model_transform, payload);
}

__device__ bool IntegratorPipeline::IntersectP(const IntegratorGlobals& globals, const Ray& ray)
{
	return false;
}

__device__ bool IntegratorPipeline::Unoccluded(const IntegratorGlobals& globals, const Ray& ray)
{
	return !(IntegratorPipeline::IntersectP(globals, ray));
}

__device__ float3 IntegratorPipeline::Li(const IntegratorGlobals& globals, const Ray& ray, uint32_t seed, float2 ppixel)
{
	return IntegratorPipeline::LiRandomWalk(globals, ray, seed, ppixel);
}

__device__ float3 SkyShading(const Ray& ray) {
	float3 unit_direction = normalize(ray.getDirection());
	float a = 0.5f * (unit_direction.y + 1.0);
	//return make_float3(0.2f, 0.3f, 0.4f);
	return (1.0f - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0);
};

__device__ float3 IntegratorPipeline::LiRandomWalk(const IntegratorGlobals& globals, const Ray& in_ray, uint32_t seed, float2 ppixel)
{
	Ray ray = in_ray;

	float3 throughtput = make_float3(1);
	float3 light = make_float3(0);

	ShapeIntersection payload;

	for (int bounce_depth = 0; bounce_depth < globals.IntegratorCFG.bounces; bounce_depth++) {
		seed += bounce_depth;
		payload = Intersect(globals, ray);

		//miss--
		if (payload.hit_distance < 0)
		{
			if (bounce_depth == 0) {
				surf2Dwrite(make_float4(0, 0, 0, 1),
					globals.FrameBuffer.normals_render_surface_object,
					ppixel.x * (int)sizeof(float4), ppixel.y);
				surf2Dwrite(make_float4(0, 0, 0.5, 1),
					globals.FrameBuffer.positions_render_surface_object,
					ppixel.x * (int)sizeof(float4), ppixel.y);
			}
			light += SkyShading(ray) * throughtput;
			break;
		}
		//hit--
		if (bounce_depth == 0) {
			//primary hit
			surf2Dwrite(make_float4(payload.w_norm, 1),
				globals.FrameBuffer.normals_render_surface_object,
				ppixel.x * (int)sizeof(float4), ppixel.y);
			surf2Dwrite(make_float4(payload.w_pos, 1),
				globals.FrameBuffer.positions_render_surface_object,
				ppixel.x * (int)sizeof(float4), ppixel.y);
		}
		light = (make_float3(1, 0, 1)); break;

		float3 wo = -ray.getDirection();
		light += payload.Le() * throughtput;

		//get BSDF
		BSDF bsdf = payload.getBSDF();

		//sample dir
		float3 wi = Samplers::sampleCosineWeightedHemisphere(payload.w_norm, Samplers::get2D_PCGHash(seed));

		float3 fcos = bsdf.f(wo, wi) * AbsDot(wi, payload.w_norm);
		if (!fcos)break;

		float pdf = 1 / (2 * PI);
		pdf = AbsDot(payload.w_norm, wi) / PI;

		throughtput *= fcos / pdf;

		ray = payload.spawnRay(wi);
	}

	return light;
};