/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   sdf.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/gpu_memory.h>

#include <vector>


namespace ngp {

struct SdfPayload {
	vec3 dir;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysSdfSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(uint32_t n_elements, const RaysSdfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(pos, other.pos, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(normal, other.normal, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(distance, other.distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(prev_distance, other.prev_distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(total_distance, other.total_distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(min_visibility, other.min_visibility, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, n_elements * sizeof(SdfPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec3* pos, vec3* normal, float* distance, float* prev_distance, float* total_distance, float* min_visibility, SdfPayload* payload) {
		this->pos = pos;
		this->normal = normal;
		this->distance = distance;
		this->prev_distance = prev_distance;
		this->total_distance = total_distance;
		this->min_visibility = min_visibility;
		this->payload = payload;
	}

	vec3* pos;
	vec3* normal;
	float* distance;
	float* prev_distance;
	float* total_distance;
	float* min_visibility;
	SdfPayload* payload;
};

struct DiscreteDistribution {
	void build(std::vector<float> weights) {
		float total_weight = 0;
		for (float w : weights) {
			total_weight += w;
		}
		float inv_total_weight = 1 / total_weight;

		float cdf_accum = 0;
		cdf.clear();
		for (float w : weights) {
			float norm = w * inv_total_weight;
			cdf_accum += norm;
			pmf.emplace_back(norm);
			cdf.emplace_back(cdf_accum);
		}
		cdf.back() = 1.0f; // Prevent precision problems from causing overruns in the end
	}

	uint32_t sample(float val) {
		return std::min(binary_search(val, cdf.data(), (uint32_t)cdf.size()), (uint32_t)cdf.size()-1);
	}

	std::vector<float> pmf;
	std::vector<float> cdf;
};

struct Triangle;
class TriangleOctree;
class TriangleBvh;


struct Sdf {
		float shadow_sharpness = 2048.0f;
		float maximum_distance = 0.00005f;
		float fd_normals_epsilon = 0.0005f;

		ESDFGroundTruthMode groundtruth_mode = ESDFGroundTruthMode::RaytracedMesh;

		BRDFParams brdf;

		// Mesh data
		EMeshSdfMode mesh_sdf_mode = EMeshSdfMode::Raystab;
		float mesh_scale;

		GPUMemory<Triangle> triangles_gpu;
		std::vector<Triangle> triangles_cpu;
		std::vector<float> triangle_weights;
		DiscreteDistribution triangle_distribution;
		GPUMemory<float> triangle_cdf;
		std::shared_ptr<TriangleBvh> triangle_bvh; // unique_ptr

		bool uses_takikawa_encoding = false;
		bool use_triangle_octree = false;
		int octree_depth_target = 0; // we duplicate this state so that you can waggle the slider without triggering it immediately
		std::shared_ptr<TriangleOctree> triangle_octree;

		GPUMemory<float> brick_data;
		uint32_t brick_res = 0;
		uint32_t brick_level = 10;
		uint32_t brick_quantise_bits = 0;
		bool brick_smooth_normals = false; // if true, then we space the central difference taps by one voxel

		bool analytic_normals = false;
		float zero_offset = 0;
		float distance_scale = 0.95f;

		double iou = 0.0;
		float iou_decay = 0.0f;
		bool calculate_iou_online = false;
		GPUMemory<uint32_t> iou_counter;
		struct Training {
			size_t idx = 0;
			size_t size = 0;
			size_t max_size = 1 << 24;
			bool did_generate_more_training_data = false;
			bool generate_sdf_data_online = true;
			float surface_offset_scale = 1.0f;
			GPUMemory<vec3> positions;
			GPUMemory<vec3> positions_shuffled;
			GPUMemory<float> distances;
			GPUMemory<float> distances_shuffled;
			GPUMemory<vec3> perturbations;
		} training = {};
};


}
