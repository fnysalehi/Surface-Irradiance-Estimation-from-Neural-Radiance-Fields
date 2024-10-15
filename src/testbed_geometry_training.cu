/** @file   testbed_geometry.cu
 *  @author Fatemeh Salehi
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/geometry.h>
#include <neural-graphics-primitives/tinyobj_loader_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/geometry_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

namespace ngp {

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

Testbed::NetworkDims Testbed::network_dims_geometry() const {
	NetworkDims dims;
	dims.n_input = 3;
	dims.n_output = 1;
	dims.n_pos = 3;
	return dims;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ inline float square(float x) { return x * x; }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline vec3 mix(const vec3& a, const vec3& b, float t) { return a + (b - a) * t; }

__device__ inline float SchlickFresnel(float u) {
	float m = __saturatef(1.0 - u);
	return square(square(m)) * m;
}

__device__ inline float G1(float NdotH, float a) {
	if (a >= 1.0) { return 1.0 / PI(); }
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return (a2 - 1.0) / (PI() * log(a2) * t);
}

__device__ inline float G2(float NdotH, float a) {
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return a2 / (PI() * t * t);
}

__device__ inline float SmithG_GGX(float NdotV, float alphaG) {
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0 / (NdotV + sqrtf(a + b - a * b));
}

__device__ vec3 evaluate_shading_geometry(
	const vec3& base_color,
	const vec3& ambient_color, // :)
	const vec3& light_color, // :)
	float metallic,
	float subsurface,
	float specular,
	float roughness,
	float specular_tint,
	float sheen,
	float sheen_tint,
	float clearcoat,
	float clearcoat_gloss,
	vec3 L,
	vec3 V,
	vec3 N
) {
	float NdotL = dot(N, L);
	float NdotV = dot(N, V);

	vec3 H = normalize(L + V);
	float NdotH = dot(N, H);
	float LdotH = dot(L, H);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	
	float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
	vec3 amb = (ambient_color * mix(0.2f, FV, metallic));
	amb *= base_color;
	if (NdotL < 0.f || NdotV < 0.f) {
		return amb;
	}

	float luminance = dot(base_color, vec3{0.3f, 0.6f, 0.1f});

	// normalize luminance to isolate hue and saturation components
	vec3 Ctint = base_color * (1.f/(luminance+0.00001f));
	vec3 Cspec0 = mix(mix(vec3(1.0f), Ctint, specular_tint) * specular * 0.08f, base_color, metallic);
	vec3 Csheen = mix(vec3(1.0f), Ctint, sheen_tint);

	float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
	float Fd = mix(1, Fd90, FL) * mix(1.f, Fd90, FV);

	// Based on Hanrahan-Krueger BRDF approximation of isotropic BSSRDF
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LdotH * LdotH * roughness;
	float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - 0.5f) + 0.5f);

	// Specular
	float a= std::max(0.001f, square(roughness));
	float Ds = G2(NdotH, a);
	float FH = SchlickFresnel(LdotH);
	vec3 Fs = mix(Cspec0, vec3(1.0f), FH);
	float Gs = SmithG_GGX(NdotL, a) * SmithG_GGX(NdotV, a);

	// sheen
	vec3 Fsheen = FH * sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = G1(NdotH, mix(0.1f, 0.001f, clearcoat_gloss));
	float Fr = mix(0.04f, 1.0f, FH);
	float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

	float CCs=0.25f * clearcoat * Gr * Fr * Dr;
	vec3 brdf = (float(1.0f / PI()) * mix(Fd, ss, subsurface) * base_color + Fsheen) * (1.0f - metallic) +
		Gs * Fs * Ds + vec3{CCs, CCs, CCs};
	return vec3(brdf * light_color) * NdotL + amb;

}

__global__ void advance_pos_kernel_mesh_geometry(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ payloads,
	BoundingBox aabb,
	float floor_y,
	float distance_scale,
	float maximum_distance,
	float k,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];
	if (!payload.alive) {
		return;
	}

	float distance = distances[i] - zero_offset;

	distance *= distance_scale;

	// Advance by the predicted distance
	vec3 pos = positions[i];
	pos += distance * payload.dir;

	if (pos.y < floor_y && payload.dir.y<0.f) {
		float floor_dist = -(pos.y-floor_y)/payload.dir.y;
		distance += floor_dist;
		pos += floor_dist * payload.dir;
		payload.alive=false;
	}

	positions[i] = pos;

	if (total_distances && distance > 0.0f) {
		// From https://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
		float total_distance = total_distances[i];
		float y = distance*distance / (2.0f * prev_distances[i]);
		float d = sqrtf(distance*distance - y*y);

		min_visibility[i] = fminf(min_visibility[i], k * d / fmaxf(0.0f, total_distance - y));
		prev_distances[i] = distance;
		total_distances[i] = total_distance + distance;
	}

	bool stay_alive = distance > maximum_distance && fabsf(distance / 2) > 3*maximum_distance;
	if (!stay_alive) {
		payload.alive = false;
		return;
	}

	if (!aabb.contains(pos)) {
		payload.alive = false;
		return;
	}

	payload.n_steps++;
}

__global__ void perturb_mesh_samples(uint32_t n_elements, const vec3* __restrict__ perturbations, vec3* __restrict__ positions, float* __restrict__ distances) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	vec3 perturbation = perturbations[i];
	positions[i] += perturbation;

	// Small epsilon above 1 to ensure a triangle is always found.
	distances[i] = length(perturbation) * 1.001f;
}

__global__ void prepare_shadow_rays_geometry(const uint32_t n_elements,
	vec3 sun_dir,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility,
	GeometryPayload* __restrict__ payloads,
	BoundingBox aabb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];

	// Step back a little along the ray to prevent self-intersection
	vec3 view_pos = positions[i] + normalize(faceforward(normals[i], payload.dir, normals[i])) * 1e-3f;
	vec3 dir = normalize(sun_dir);

	float t = fmaxf(aabb.ray_intersect(view_pos, dir).x + 1e-6f, 0.0f);
	view_pos += t * dir;

	positions[i] = view_pos;

	if (!aabb.contains(view_pos)) {
		distances[i] = MAX_DEPTH();
		payload.alive = false;
		min_visibility[i] = 1.0f;
		return;
	}

	distances[i] = MAX_DEPTH();
	payload.idx = i;
	payload.dir = dir;
	payload.n_steps = 0;
	payload.alive = true;

	if (prev_distances) {
		prev_distances[i] = 1e20f;
	}

	if (total_distances) {
		total_distances[i] = 0.0f;
	}

	if (min_visibility) {
		min_visibility[i] = 1.0f;
	}
}

__global__ void prepare_shadow_rays_envmap_geometry(const uint32_t n_elements,
	vec3 center,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility,
	GeometryPayload* __restrict__ payloads,
	BoundingBox aabb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];

	// Step back a little along the ray to prevent self-intersection
	vec3 view_pos = positions[i] + normalize(faceforward(normals[i], payload.dir, normals[i])) * 1e-3f;

	vec3 dir = normalize(center - view_pos);

	float t = fmaxf(aabb.ray_intersect(view_pos, dir).x + 1e-6f, 0.0f);
	view_pos += t * dir;

	positions[i] = view_pos;

	if (!aabb.contains(view_pos)) {
		// printf("t: %f \n", t);
		distances[i] = MAX_DEPTH();
		payload.alive = false;
		min_visibility[i] = 1.0f;
		return;
	}

	distances[i] = MAX_DEPTH();
	payload.idx = i;
	payload.dir = dir;
	payload.n_steps = 0;
	payload.alive = true;

	if (prev_distances) {
		prev_distances[i] = 1e20f;
	}

	if (total_distances) {
		total_distances[i] = 0.0f;
	}

	if (min_visibility) {
		min_visibility[i] = 1.0f;
	}
}



__global__ void write_shadow_ray_result_geometry(const uint32_t n_elements, BoundingBox aabb, const vec3* __restrict__ positions, const GeometryPayload* __restrict__ shadow_payloads, const float* __restrict__ min_visibility, float* __restrict__ shadow_factors) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	shadow_factors[shadow_payloads[i].idx] = aabb.contains(positions[i]) ? 0.0f : min_visibility[i];
}

__global__ void write_shadow_ray_result_envmap(
    const uint32_t n_elements,
    GeometryPayload* __restrict__ payloads,
	vec3* __restrict__ positions,
    float* __restrict__ weights,
    cudaTextureObject_t* envmap,
    uint32_t envmap_width,
    uint32_t envmap_height,
    float* __restrict__ result
) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= n_elements) {
        return;
    }

    GeometryPayload& payload = payloads[idx];
    float weight = weights[idx];
	vec3 pos = positions[idx];

    if (!payload.alive) {
        return;
    }

    // Transform the position of the ray to UV coordinates
    vec2 uv_pos = dir_to_spherical_unorm(pos);

    // Calculate an index based on the UV coordinates
    uint32_t index = uv_pos.x * envmap_width + uv_pos.y * envmap_height;

    // Transform the direction of the ray to UV coordinates
    vec2 uv_dir = dir_to_spherical_unorm(payload.dir);

    // Fetch the environment map
    float4 envmap_value = tex3D<float4>(*envmap, uv_dir.x, uv_dir.y, index);

    // Write the result
    result[index] = weight * envmap_value.x;
}

__global__ void write_shadow_ray_result_from_nerf(
	const uint32_t n_hit_nerf_rays, 
	vec4* __restrict__ rgba,
	NerfPayload* __restrict__ payloads,
	size_t n_rays_per_sample,
	vec3* __restrict__ rgb,
	float* __restrict__ weights

) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_hit_nerf_rays) return;

	NerfPayload& payload = payloads[i];

	vec3 color =  rgba[i].rgb();

	// sum f/p = color * ndotl/pi / pdf = color
	// sum f/p = color * ndotl/pi * area          # area = 1/pdf

	float weight = weights[payload.idx];
	rgb[payload.idx] += vec3(weight * color[0]/ n_rays_per_sample, weight * color[1]/ n_rays_per_sample, weight * color[2]/ n_rays_per_sample);
}

__global__ void write_shadow_ray_result_from_nerf_envmap_grid(
	const uint32_t n_hit_rays, 
	// BoundingBox aabb,
	vec3* __restrict__ positions,
	GeometryPayload* __restrict__ payloads,
	float* __restrict__ min_visibility,
	size_t n_rays_per_sample,
	vec3* __restrict__ rgb,
	float* __restrict__ weights,
	const ivec2& gridSize,
	cudaTextureObject_t* envmapTex

) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_hit_rays) return;

	GeometryPayload& payload = payloads[i];
	// Calculate the cell indices from the direction vector

	vec3 pos = positions[payload.idx];
	vec2 uv_pos = dir_to_spherical(pos);

    // Calculate an index based on the UV coordinates
    uint32_t index = uv_pos.y * gridSize.y * gridSize.x + uv_pos.x * gridSize.y;


    // Transform the direction of the ray to UV coordinates
	// maybe minus
	// printf("payload.dir: %f %f %f \n", payload.dir.x, payload.dir.y, payload.dir.z);
    vec2 uv_dir = dir_to_spherical(-payload.dir);
	// printf("uv_dir: %f %f %d \n", uv_dir.x, uv_dir.y, index);
    // Fetch the environment map
    float4 rgba = tex2DLayered<float4>(*envmapTex, uv_dir.x, uv_dir.y, index);
	// printf("envmap_value: %f %f %f %f \n", rgba.x, rgba.y, rgba.z, rgba.w);
    // Write the result
	
	vec3 color = vec3(rgba.x, rgba.y, rgba.z);

	// sum f/p = color * ndotl/pi / pdf = color
	// sum f/p = color * ndotl/pi * area          # area = 1/pdf

	float weight = weights[payload.idx];
	rgb[payload.idx] += vec3(weight * color[0]/ n_rays_per_sample, weight * color[1]/ n_rays_per_sample, weight * color[2]/ n_rays_per_sample);
	// if(color[0] > 0.0f || color[1] > 0.0f || color[2] > 0.0f)
	// 	printf("color: %f %f %f \n", color[0], color[1], color[2]);
	// printf("color: %f %f %f \n", color[0], color[1], color[2]);
}

__global__ void setVec3(const uint32_t n_elements, vec3* __restrict__ rgb, float value) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n_elements) {
        rgb[i] = vec3(value, value, value);
    }
}

__global__ void write_envmap_ray_result_from_nerf(
	const uint32_t n_hit_nerf_rays, 
	vec4* __restrict__ rgbas,
	NerfPayload* __restrict__ payloads,
	vec4* __restrict__ envmap
) {
	
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_hit_nerf_rays) return;

	NerfPayload& payload = payloads[i];
	vec4 rgba = rgbas[i];
	// envmap[payload.idx] = linear_to_srgb(rgba);
	envmap[payload.idx] = rgba;
	// if(rgba[0] > 0.0f || rgba[1] > 0.0f || rgba[2] > 0.0f)
	// 	printf("rgba: %f %f %f \n", rgba[0], rgba[1], rgba[2]);
	// 	vec2 uv_dir = dir_to_spherical(payload.dir);
	// 	printf("uv_dir: %f %f \n", uv_dir.x, uv_dir.y);
    
		// printf("Direction: (%f, %f, %f)\n", payload.dir.x, payload.dir.y, payload.dir.z);
}

__global__ void write_envmap_multiple_ray_result_from_nerf(
	const uint32_t n_hit_nerf_rays, 
	uint32_t numSamplesOrigin,
	vec4* __restrict__ rgbas,
	NerfPayload* __restrict__ payloads,
	vec4* __restrict__ envmap
) {
	
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_hit_nerf_rays) return;

	NerfPayload& payload = payloads[i];
	
	vec4 rgba = rgbas[i];
	uint32_t numSamples = numSamplesOrigin * numSamplesOrigin;
	// atomic add instead of this to be sure
	envmap[payload.idx] += vec4(rgba[0] / numSamples, rgba[1] / numSamples, rgba[2] / numSamples, rgba[3] / numSamples);

}

__global__ void write_rgb_from_nerf(
	const uint32_t n_hit_nerf_rays, 
	vec4* __restrict__ rgba,
	NerfPayload* __restrict__ payloads,
	size_t n_rays_per_sample,
	vec3* __restrict__ rgb_out

) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_hit_nerf_rays) return;

	NerfPayload& payload = payloads[i];
	
	vec3 color = rgba[i].rgb();
	// printf("color: %f %f %f \n", color[0], color[1], color[2]);
	rgb_out[payload.idx] += vec3(100*color[0]/ n_rays_per_sample, 100*color[1]/ n_rays_per_sample, 100*color[2]/ n_rays_per_sample);
}

// __global__ void shade_kernel_from_nerf_oldv(
// 	const uint32_t n_elements,
// 	BoundingBox aabb,
// 	float floor_y,
// 	const ERenderMode mode,
// 	const BRDFParams brdf,
// 	vec3 up_dir,
// 	mat4x3 camera_matrix,
// 	vec3* __restrict__ positions,
// 	vec3* __restrict__ normals,
// 	float* __restrict__ distances,
// 	GeometryPayload* __restrict__ original_payloads,
// 	vec4* __restrict__ frame_buffer,
// 	float* __restrict__ depth_buffer,

// 	BoundingBox nerf_aabb,
// 	vec4* __restrict__ rgba,
// 	float* __restrict__ depth,
// 	NerfPayload* __restrict__ payloads,
// 	const uint32_t n_rays_per_sample,
// 	const ivec2& resolution

// ) {
	
// 	// we have n_hit * n_rays_per_sample nerf rays
// 	// we have n_hit mesh rays
// 	printf("beginning of shade_kernel_from_nerf \n");

// 	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (i >= n_elements ) return;
	
// 	NerfPayload& payload = payloads[i];
	
// 	GeometryPayload& original_payload = original_payloads[payload.idx];
	
// 	// The normal in memory isn't normalized yet
// 	vec3 normal = normalize(normals[payload.idx]);
// 	printf("after check \n");
// 	vec3 pos = positions[payload.idx];
	
// 	if (!aabb.contains(pos)) {
// 		return;
// 	}
	
// 	bool floor = false;
// 	if (pos.y < floor_y + 0.001f && original_payload.dir.y < 0.f) {
// 		normal = vec3{0.0f, 1.0f, 0.0f};
// 		floor = true;
// 	}
	
// 	vec3 cam_pos = camera_matrix[3];
// 	vec3 cam_fwd = camera_matrix[2];
// 	float ao = powf(0.92f, original_payload.n_steps * 0.5f) * (1.f / 0.92f);
// 	vec4 tmp = rgba[i];
	
// 	float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
// 	const vec3 skycol = vec3{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
// 	float check_size = 8.f/aabb.diag().x;
// 	float check=((int(floorf(check_size*(pos.x-aabb.min.x)))^int(floorf(check_size*(pos.z-aabb.min.z)))) &1) ? 0.8f : 0.2f;
// 	const vec3 floorcol = vec3{check*check*check, check*check, check};
	
// 	vec3 color = evaluate_shading_geometry(
// 		floor ? floorcol : brdf.basecolor * brdf.basecolor,
// 		brdf.ambientcolor * skycol,
// 		tmp.rgb(),	// light color
// 		floor ? 0.f : brdf.metallic,
// 		floor ? 0.f : brdf.subsurface,
// 		floor ? 1.f : brdf.specular,
// 		floor ? 0.5f : brdf.roughness,
// 		0.f,
// 		floor ? 0.f : brdf.sheen,
// 		0.f,
// 		floor ? 0.f : brdf.clearcoat,
// 		brdf.clearcoat_gloss,
// 		payload.dir,	//from the point toward the light source (L light)
// 		-normalize(original_payload.dir),	//from the mesh surface to the camera (v view)
// 		normal	// normal
// 	);
	
// 	frame_buffer[original_payload.idx] += {color[0]/ n_rays_per_sample, color[1]/ n_rays_per_sample, color[2]/ n_rays_per_sample, 0.0f};
// 	depth_buffer[original_payload.idx] = dot(cam_fwd, pos - cam_pos);
// }

__global__ void shade_kernel_from_nerf(
	const uint32_t n_elements,
	BoundingBox aabb,
	float floor_y,
	const ERenderMode mode,
	const BRDFParams brdf,
	vec3 up_dir,
	mat4x3 camera_matrix,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ original_payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,

	BoundingBox nerf_aabb,
	vec3* __restrict__ rgb,
	NerfPayload* __restrict__ payloads,
	const uint32_t n_rays_per_sample

) {
	
	// we have n_hit * n_rays_per_sample nerf rays
	// we have n_hit mesh rays

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements ) return;
		
	GeometryPayload& original_payload = original_payloads[i];
	
	vec3 normal = normalize(normals[i]);
	vec3 pos = positions[i];
	
	if (!aabb.contains(pos)) {
		return;
	}
		
	vec3 color = linear_to_srgb(rgb[original_payload.idx]);
	vec3 cam_pos = camera_matrix[3];
	vec3 cam_fwd = camera_matrix[2];

	frame_buffer[original_payload.idx] = {color[0] , color[1], color[2], 1.0f};
	depth_buffer[original_payload.idx] = dot(cam_fwd, pos - cam_pos);

}


__global__ void shade_kernel_from_nerf_post(
	const uint32_t n_elements,
	vec4* __restrict__ frame_buffer

) {
	
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements ) return;
	
	// frame_buffer[i] = linear_to_srgb(frame_buffer[i].rgb());
	// frame_buffer[i].a = 1.0f;

}

inline __host__ __device__ vec2   dir_to_cylindrical_nerf(const vec3& d) {
	const float cos_theta = fminf(fmaxf(-d.z, -1.0f), 1.0f);
	float phi = atan2(d.y, d.x);
	// printf("cos_theta: %f, phi: %f\n", (cos_theta + 1.0f) / 2.0f, (phi / (2.0f * PI())) + 0.5f);
	return {(cos_theta + 1.0f) / 2.0f, (phi / (2.0f * PI())) + 0.5f};
}


__global__ void shade_kernel_mesh_geometry(
	const uint32_t n_elements,
	BoundingBox aabb,
	float floor_y,
	const ERenderMode mode,
	const BRDFParams brdf,
	vec3 sun_dir,
	vec3 up_dir,
	mat4x3 camera_matrix,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	cudaTextureObject_t* envmapTex,
	vec3 nerfCenter = vec3(0.0f, 0.0f, 0.0f),
	vec2 cellSize = vec2(0.0f, 0.0f),
	vec3* __restrict__ rgb = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];
	if (!aabb.contains(positions[i])) {
		return;
	}

	// The normal in memory isn't normalized yet
	vec3 normal = normalize(normals[i]);
	vec3 pos = positions[i];
	bool floor = false;
	if (pos.y < floor_y + 0.001f && payload.dir.y < 0.f) {
		normal = vec3{0.0f, 1.0f, 0.0f};
		floor = true;
	}

	vec3 cam_pos = camera_matrix[3];
	vec3 cam_fwd = camera_matrix[2];
	float ao = powf(0.92f, payload.n_steps * 0.5f) * (1.f / 0.92f);
	vec3 color;
	switch (mode) {
		case ERenderMode::AO: color = vec3(powf(0.92f, payload.n_steps)); break;
		case ERenderMode::Shade: {
			float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
			vec3 suncol = vec3{255.f/255.0f, 225.f/255.0f, 195.f/255.0f} * 4.f * distances[i]; // Distance encodes shadow occlusion. 0=occluded, 1=no shadow
			const vec3 skycol = vec3{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
			float check_size = 8.f/aabb.diag().x;
			float check=((int(floorf(check_size*(pos.x-aabb.min.x)))^int(floorf(check_size*(pos.z-aabb.min.z)))) &1) ? 0.8f : 0.2f;
			const vec3 floorcol = vec3{check*check*check, check*check, check};
			color = evaluate_shading_geometry(
				floor ? floorcol : brdf.basecolor * brdf.basecolor,
				brdf.ambientcolor * skycol,
				suncol,
				floor ? 0.f : brdf.metallic,
				floor ? 0.f : brdf.subsurface,
				floor ? 1.f : brdf.specular,
				floor ? 0.5f : brdf.roughness,
				0.f,
				floor ? 0.f : brdf.sheen,
				0.f,
				floor ? 0.f : brdf.clearcoat,
				brdf.clearcoat_gloss,
				sun_dir,
				-normalize(payload.dir),
				normal
			);
		} break;
		case ERenderMode::Depth: color = vec3(dot(cam_fwd, pos - cam_pos)); break;
		case ERenderMode::Positions: {
			color = (pos - 0.5f) / 2.0f + 0.5f;
		} break;
		case ERenderMode::Normals: color = 0.5f * normal + 0.5f; break;
		case ERenderMode::Cost: color = vec3((float)payload.n_steps / 30); break;
		case ERenderMode::EncodingVis: color = normals[i]; break;

		case ERenderMode::ShadeEnvMap: {
			vec3 dir = normalize(pos - nerfCenter);
			auto thetaphi = dir_to_cylindrical_nerf(dir);
			// auto thetaphi = dir_to_spherical_unorm(dir);
			
			// printf("thetaphi: %f %f \n", thetaphi.x, thetaphi.y);
			// printf("dir from shading: %f %f %f \n", dir.x, dir.y, dir.z);
			//the problem line
			float4 rgba= make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			rgba = tex2D<float4>(*envmapTex, thetaphi.x, thetaphi.y);
			// printf("rgba: %f %f %f \n", rgba.x, rgba.y, rgba.z);
			vec3 diff = pos - nerfCenter;
			float magnitude = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z; //sqrt
			
			vec3 nerfcolor = vec3(rgba.x, rgba.y, rgba.z);
			nerfcolor /= magnitude;
			nerfcolor = linear_to_srgb(nerfcolor);
			
			// if(nerfcolor.x > 0.0f || nerfcolor.y > 0.0f || nerfcolor.z > 0.0f)
			// 	printf("nerfcolor: %f %f %f \n", nerfcolor.x, nerfcolor.y, nerfcolor.z);
			// printf("nerfcolor: %f %f %f \n", rgba.x, rgba.y, rgba.z);
			
			float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
			const vec3 skycol = vec3{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
			float check_size = 8.f/aabb.diag().x;
			float check=((int(floorf(check_size*(pos.x-aabb.min.x)))^int(floorf(check_size*(pos.z-aabb.min.z)))) &1) ? 0.8f : 0.2f;
			const vec3 floorcol = vec3{check*check*check, check*check, check};
			
			
			color = evaluate_shading_geometry(
				floor ? floorcol : brdf.basecolor * brdf.basecolor,
				brdf.ambientcolor * skycol,
				nerfcolor,
				floor ? 0.f : brdf.metallic,
				floor ? 0.f : brdf.subsurface,
				floor ? 1.f : brdf.specular,
				floor ? 0.5f : brdf.roughness,
				0.f,
				floor ? 0.f : brdf.sheen,
				0.f,
				floor ? 0.f : brdf.clearcoat,
				brdf.clearcoat_gloss,
				-dir,
				-normalize(payload.dir),
				normal
			);
		} break;
		case ERenderMode::ShadeGridEnvMap: {

			// Calculate the cell indices from the direction vector
			
			vec3 nerfcolor = rgb[i];

			vec3 dir = normalize(pos - nerfCenter);
			
			// Calculate the relative position within the bounding box
			vec3 relPos = (pos - aabb.min) / aabb.diag();

			vec3 diff = pos - nerfCenter;
			float magnitude = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z; //sqrt
			
			nerfcolor /= magnitude;
			nerfcolor = linear_to_srgb(nerfcolor);
			
			// if(nerfcolor.x > 0.0f || nerfcolor.y > 0.0f || nerfcolor.z > 0.0f)
			// 	printf("nerfcolor: %f %f %f \n", nerfcolor.x, nerfcolor.y, nerfcolor.z);
			// printf("nerfcolor: %f %f %f \n", rgba.x, rgba.y, rgba.z);
			
			float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
			const vec3 skycol = vec3{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
			float check_size = 8.f/aabb.diag().x;
			float check=((int(floorf(check_size*(pos.x-aabb.min.x)))^int(floorf(check_size*(pos.z-aabb.min.z)))) &1) ? 0.8f : 0.2f;
			const vec3 floorcol = vec3{check*check*check, check*check, check};
			
			
			color = evaluate_shading_geometry(
				floor ? floorcol : brdf.basecolor * brdf.basecolor,
				brdf.ambientcolor * skycol,
				nerfcolor,
				floor ? 0.f : brdf.metallic,
				floor ? 0.f : brdf.subsurface,
				floor ? 1.f : brdf.specular,
				floor ? 0.5f : brdf.roughness,
				0.f,
				floor ? 0.f : brdf.sheen,
				0.f,
				floor ? 0.f : brdf.clearcoat,
				brdf.clearcoat_gloss,
				-dir,
				-normalize(payload.dir),
				normal
			);
		} break;

	}

	frame_buffer[payload.idx] = {color.r, color.g, color.b, 1.0f};
	depth_buffer[payload.idx] = dot(cam_fwd, pos - cam_pos);
}

__global__ void shade_kernel_mesh_from_nerf_geometry(
	const uint32_t n_elements,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer

	) {
	
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) 
		return;
	
	// NerfPayload& payload = payloads[i];
	// vec4 tmp = rgba[i];

    // // Compute shading from the NeRF
    // vec3 color_nerf = vec3(0.0f);
	// vec3 incoming_light = vec3(0.0f);
    // for (int sample = 0; sample < n_samples; ++sample) {
    //     vec3 sample_dir = random_dir_cosine(rng);
	// 	vec4 volume_density_and_color = nerf->evaluate(pos, sample_dir);
	// 	color_nerf += vec3(volume_density_and_color);
    // }
    // color_nerf /= n_samples;

	// // Convert the color to sRGB
    // tmp.rgb() = linear_to_srgb(color_nerf);

    // // Write the final color to the frame buffer
    // rgba[i] = tmp;

}
__global__ void compact_kernel_shadow_mesh_geometry(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* src_positions, float* src_distances, GeometryPayload* src_payloads, float* src_prev_distances, float* src_total_distances, float* src_min_visibility,
	vec3* dst_positions, float* dst_distances, GeometryPayload* dst_payloads, float* dst_prev_distances, float* dst_total_distances, float* dst_min_visibility,
	vec3* dst_final_positions, float* dst_final_distances, GeometryPayload* dst_final_payloads, float* dst_final_prev_distances, float* dst_final_total_distances, float* dst_final_min_visibility,
	BoundingBox aabb,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
		dst_prev_distances[idx] = src_prev_distances[i];
		dst_total_distances[idx] = src_total_distances[i];
		dst_min_visibility[idx] = src_min_visibility[i];
	} else { // For shadow rays, collect _all_ final samples to keep track of their partial visibility
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = src_distances[i];
		dst_final_prev_distances[idx] = src_prev_distances[i];
		dst_final_total_distances[idx] = src_total_distances[i];
		dst_final_min_visibility[idx] = aabb.contains(src_positions[i]) ? 0.0f : src_min_visibility[i];
	}
}

// separates the "alive" and "dead" elements of the input arrays into two separate arrays
__global__ void compact_kernel_mesh_geometry(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* src_positions, float* src_distances, GeometryPayload* src_payloads,
	vec3* dst_positions, float* dst_distances, GeometryPayload* dst_payloads,
	vec3* dst_final_positions, float* dst_final_distances, GeometryPayload* dst_final_payloads,
	BoundingBox aabb,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
	} else if (aabb.contains(src_positions[i])) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = 1.0f; // HACK: Distances encode shadowing factor when shading
	}
}

__global__ void uniform_octree_sample_kernel_geometry(
	const uint32_t num_elements,
	default_rng_t rng,
	const TriangleOctreeNode* __restrict__ octree_nodes,
	uint32_t num_nodes,
	uint32_t depth,
	vec3* __restrict__ samples
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_elements) return;

	rng.advance(i * (1<<8));

	// Samples random nodes until a leaf is picked
	uint32_t node;
	uint32_t child;
	do {
		node = umin((uint32_t)(random_val(rng) * num_nodes), num_nodes-1);
		child = umin((uint32_t)(random_val(rng) * 8), 8u-1);
	} while (octree_nodes[node].depth < depth-2 || octree_nodes[node].children[child] == -1);

	// Here it should be guaranteed that any child of the node is -1
	float size = scalbnf(1.0f, -depth+1);

	u16vec3 pos = octree_nodes[node].pos * uint16_t(2);
	if (child&1) ++pos.x;
	if (child&2) ++pos.y;
	if (child&4) ++pos.z;
	samples[i] = size * (vec3(pos) + samples[i]);
}

__global__ void scale_to_aabb_kernel_geometry(uint32_t n_elements, BoundingBox aabb, vec3* __restrict__ inout) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	inout[i] = aabb.min + inout[i] * aabb.diag();
}

__global__ void compare_signs_kernel_geometry(uint32_t n_elements, const vec3 *positions, const float *distances_ref, const float *distances_model, uint32_t *counters) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	bool inside1 = distances_ref[i]<=0.f;
	bool inside2 = distances_model[i]<=0.f;
	
	// atomicAdd(&counters[7],1);

	atomicAdd(&counters[inside1 ? 0 : 1],1);
	atomicAdd(&counters[inside2 ? 2 : 3],1);
	if (inside1&&inside2)
		atomicAdd(&counters[4],1);
	if (inside1||inside2)
		atomicAdd(&counters[5],1);

}

__global__ void assign_float_geometry(uint32_t n_elements, float value, float* __restrict__ out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	out[i] = value;
}

__global__ void init_rays_with_payload_kernel_mesh_geometry(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask
	);

	distances[idx] = MAX_DEPTH();
	depth_buffer[idx] = MAX_DEPTH();

	GeometryPayload& payload = payloads[idx];

	if (!ray.is_valid()) {
		payload.dir = ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.dir = (1.0f/n) * ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o - plane_z * ray.d;
		depth_buffer[idx] = -plane_z;
		return;
	}

	ray.d = normalize(ray.d);
	float t = max(aabb.ray_intersect(ray.o, ray.d).x, 0.0f);

	ray.advance(t + 1e-6f);

	positions[idx] = ray.o;

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

	payload.dir = ray.d;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = aabb.contains(ray.o);
}


__global__ void init_rays_with_payload_kernel_nerf_geometry(
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	NerfPayload& payload = payloads[idx];
	payload.max_weight = 0.0f;

	if (depth_buffer == nullptr || depth_buffer[idx] < 0.01) {

		depth_buffer[idx] = MAX_DEPTH();
	}

	if (!ray.is_valid()) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.origin = ray.o;
		payload.dir = (1.0f/n) * ray.d;
		payload.t = -plane_z*n;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		depth_buffer[idx] = -plane_z;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		vec2 uv_after_distortion = pos_to_uv(ray(1.0f), resolution, focal_length, camera, screen_center, parallax_shift, foveation);

		frame_buffer[idx].rgb() = to_rgb((uv_after_distortion - uv) * 64.0f);
		frame_buffer[idx].a = 1.0f;
		depth_buffer[idx] = 1.0f;
		payload.origin = ray(MAX_DEPTH());
		payload.alive = false;
		return;
	}

	ray.d = normalize(ray.d);

	float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	if (!render_aabb.contains(render_aabb_to_local * ray(t))) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = t;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void grid_to_bitfield_geometry(
	const uint32_t n_elements,
	const uint32_t n_nonzero_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	if (i >= n_nonzero_elements) {
		grid_bitfield[i] = 0;
		return;
	}

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool_geometry(const uint32_t n_elements,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
	uint32_t y = morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
	uint32_t z = morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

	next_level[morton3D(x, y, z)] |= bits;
}

__global__ void generate_nerf_network_inputs_at_current_position_geometry(const uint32_t n_elements, BoundingBox aabb, const NerfPayload* __restrict__ payloads, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	vec3 dir = payloads[i].dir;
	network_input(i)->set_with_optional_extra_dims(warp_position(payloads[i].origin + dir * payloads[i].t, aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__device__ vec4 compute_nerf_rgba_geometry(const vec4& network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
	vec4 rgba = network_output;

	float density = network_to_density(rgba.a, density_activation);
	float alpha = 1.f;
	if (density_as_alpha) {
		rgba.a = density;
	} else {
		rgba.a = alpha = clamp(1.f - __expf(-density * depth), 0.0f, 1.0f);
	}

	rgba.rgb() = network_to_rgb_vec(rgba.rgb(), rgb_activation) * alpha;
	return rgba;
}

__global__ void compute_nerf_rgba_kernel_geometry(const uint32_t n_elements, vec4* network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	network_output[i] = compute_nerf_rgba_geometry(network_output[i], rgb_activation, density_activation, depth, density_as_alpha);
}

__global__ void shade_kernel_nerf_geometry(
	const uint32_t n_elements,
	bool gbuffer_hard_edges,
	mat4x3 camera_matrix,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	ERenderMode render_mode,
	bool train_in_linear_colors,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements || render_mode == ERenderMode::Distortion) return;
	NerfPayload& payload = payloads[i];
	
	if (depth[i] > depth_buffer[payload.idx]) {
        // The NeRF is further away than the mesh, so skip rendering the NeRF
        return;
    }
	
	vec4 tmp = rgba[i];
	if (render_mode == ERenderMode::Normals) {
		vec3 n = normalize(tmp.xyz());
		tmp.rgb() = (0.5f * n + 0.5f) * tmp.a;
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	} else if (gbuffer_hard_edges && render_mode == ERenderMode::Depth) {
		tmp.rgb() = vec3(depth[i] * depth_scale);
	} else if (gbuffer_hard_edges && render_mode == ERenderMode::Positions) {
		vec3 pos = camera_matrix[3] + payload.dir / dot(payload.dir, camera_matrix[2]) * depth[i];
		tmp.rgb() = (pos - 0.5f) / 2.0f + 0.5f;
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice || render_mode == ERenderMode::ShadeNerf)) {
		// Accumulate in linear colors
		tmp.rgb() = srgb_to_linear(tmp.rgb());
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);
	if (render_mode != ERenderMode::Slice && tmp.a > 0.2f) {
		depth_buffer[payload.idx] = depth[i];
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// shade
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void accumulate_samples(NerfPayload* payloads, vec4* frame_buffer, uint32_t n_samples, ivec2 resolution) {
    // uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    // uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    // if (x >= resolution.x || y >= resolution.y) {
    //     return;
    // }

    // uint32_t idx = x + resolution.x * y;

    // vec4 total_color = vec4(0.0f);
    // for (uint32_t i = 0; i < n_samples; ++i) {
    //     total_color += payloads[idx * n_samples + i].color;
    // }

    // frame_buffer[idx] = total_color / n_samples;
}

// __global__ void init_rays_with_payload_kernel_shade_geometry_oldv(
//     const uint32_t n_rays_per_sample,
//     NerfPayload* __restrict__ payloads,
//     const uint32_t n_elements,
//     vec2 screen_center,
//     vec3 parallax_shift,
//     bool snap_to_pixel_centers,
//     BoundingBox nerfaabb,
//     mat3 render_aabb_to_local,
//     vec3* __restrict__ positions,
// 	float* __restrict__ distances,
// 	GeometryPayload* __restrict__ meshpayloads,
// 	vec2* __restrict__ random_numbers,
// 	const ivec2& resolution

// ) {
// 	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
// 	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

// 	if (x >= resolution.x * n_rays_per_sample || y >= resolution.y * n_rays_per_sample ) {
// 		return;
// 	}

// 	uint32_t idx = x + resolution.x * n_rays_per_sample * y;
	
//     // uint32_t hit_ray_idx = idx / (n_rays_per_sample * n_rays_per_sample);
// 	uint32_t hit_ray_idx = x/ n_rays_per_sample + resolution.x * (y / n_rays_per_sample);

// 	// if (idx >= resolution.x * resolution.y * n_elements * n_elements) {
// 	// 	return;
// 	// }
// 	NerfPayload& payload = payloads[idx];
//     payload.max_weight = 0.0f;
	
// 	GeometryPayload meshpayload = meshpayloads[hit_ray_idx];
	
// 	// make sure the coordinate system of both is the same 
// 	vec3 view_pos = positions[hit_ray_idx] + normalize(faceforward(normals[i], payload.dir, normals[i])) * 1e-3f;
	
// 	Ray ray = {view_pos, normalize(cosine_hemisphere(random_numbers[idx]))};
	
// 	// Ray ray = {positions[hit_ray_idx], meshpayload.dir};
// 	float t = fmaxf(nerfaabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

// 	if (!nerfaabb.contains(render_aabb_to_local * ray(t))) {
// 		payload.origin = ray.o;
// 		payload.alive = false;
// 		return;
// 	}

// 	payload.origin = ray.o;
// 	payload.dir = ray.d;
// 	payload.t = t;
// 	payload.idx = hit_ray_idx;
// 	payload.n_steps = 0;
// 	payload.alive = true;

// }

__global__ void init_rays_with_payload_kernel_shade_geometry(
    const uint32_t n_elements,
    NerfPayload* __restrict__ payloads,
    const uint32_t n_rays_per_sample,
    BoundingBox nerfaabb,
    mat3 render_aabb_to_local,
    vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ meshpayloads,
	vec2* __restrict__ random_vecs,
	float* __restrict__ weights

) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements) {
		return;
	}
	
	uint32_t hit_ray_idx = idx / n_rays_per_sample;

	NerfPayload& payload = payloads[idx];
    payload.max_weight = 0.0f;
	
	GeometryPayload meshpayload = meshpayloads[hit_ray_idx];
	vec3 N = normals[hit_ray_idx];
	// make sure the coordinate system of both is the same 
	vec3 view_pos = positions[hit_ray_idx];	// + normalize(faceforward(normals[hit_ray_idx], meshpayload.dir, normals[hit_ray_idx])) * 1e-3f;
	// printf("view_pos: %f, %f, %f\n", view_pos.x, view_pos.y, view_pos.z);
	
	vec3 light_center_dir = normalize( nerfaabb.center() - view_pos);
	mat3 local_frame = compute_local_frame(light_center_dir);

	float light_center_distance_squared = length2(nerfaabb.center() - view_pos);
	float r = nerfaabb.radius();
	float cap_height = 1.0f - sqrtf((light_center_distance_squared - r * r) / light_center_distance_squared);

	vec3 local_direction_to_light = normalize(warp_square_to_spherical_cap_uniform(random_vecs[idx], cap_height));
	// vec3 dir = normalize(cosine_hemisphere(random_vecs[idx]));
    vec3 dir = local_frame * local_direction_to_light;

	Ray ray = {view_pos, dir};
	float NdotL = dot(N, dir);

	weights[hit_ray_idx] = warp_square_to_spherical_cap_uniform_invpdf(local_direction_to_light, cap_height) * NdotL / PI();
	
	float t = fmaxf(nerfaabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	if (!nerfaabb.contains(render_aabb_to_local * ray(t))) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}
	
	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = 1e-6f;
	payload.idx = meshpayload.idx;
	payload.n_steps = 0;
	payload.alive = true;

}

__global__ void init_rays_with_payload_kernel_shade_raymesh(
    const uint32_t n_elements,
    GeometryPayload* __restrict__ payloads,
	vec3* __restrict__ shading_ray_positions,
	vec3* __restrict__ shading_ray_normals,
    const uint32_t n_rays_per_sample,
    BoundingBox nerfaabb,
    mat3 render_aabb_to_local,
    vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ meshpayloads,
	vec2* __restrict__ random_vecs,
	float* __restrict__ weights

) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements) {
		return;
	}
	
	uint32_t hit_ray_idx = idx / n_rays_per_sample;

	GeometryPayload& shadepayload = payloads[idx];
	
	GeometryPayload originalpayload = meshpayloads[hit_ray_idx];
	vec3 N = normals[hit_ray_idx];
	// make sure the coordinate system of both is the same 
	vec3 view_pos = positions[hit_ray_idx];	// + normalize(faceforward(normals[hit_ray_idx], meshpayload.dir, normals[hit_ray_idx])) * 1e-3f;
	
	vec3 light_center_dir = normalize( nerfaabb.center() - view_pos);
	mat3 local_frame = compute_local_frame(light_center_dir);

	float light_center_distance_squared = length2(nerfaabb.center() - view_pos);
	float r = nerfaabb.radius();
	float cap_height = 1.0f - sqrtf((light_center_distance_squared - r * r) / light_center_distance_squared);

	vec3 local_direction_to_light = normalize(warp_square_to_spherical_cap_uniform(random_vecs[idx], cap_height));
	// vec3 dir = normalize(cosine_hemisphere(random_vecs[idx]));
    vec3 dir = local_frame * local_direction_to_light;

	Ray ray = {view_pos, dir};
	float NdotL = dot(N, dir);

	weights[hit_ray_idx] = warp_square_to_spherical_cap_uniform_invpdf(local_direction_to_light, cap_height) * NdotL / PI();
	
	// vec2 uv = dir_to_spherical_unorm(render_aabb_to_local * ray.d);
    // vec3 pointOnSphere = fmaxf(nerfaabb.ray_intersect_sphere(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f);


	// float t = fmaxf(nerfaabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;
	
	vec3 pointOnSphere = nerfaabb.ray_intersect_sphere(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x;
	// check if it is NAN
	if (!all(isfinite(pointOnSphere))) {
		shadepayload.alive = false;
		return;
	}
	else {
		
		auto point = render_aabb_to_local * pointOnSphere;
		ray.o = point;
	}
	

	shading_ray_positions[idx] = ray.o;
	shading_ray_normals[idx] = N;

	shadepayload.dir = ray.d;
	shadepayload.idx = originalpayload.idx;
	shadepayload.n_steps = 0;
	shadepayload.alive = true;

}

__global__ void compact_kernel_shade(
	const uint32_t n_elements,
	vec3* src_positions, 		vec3* src_normal, 		GeometryPayload* src_payloads,
	vec3* dst_final_positions, 	vec3* dst_final_normal, GeometryPayload* dst_final_payloads,
	uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_normal[idx] = src_normal[i];
	}
}


__global__ void visualize_t(
    const uint32_t n_elements,

	GeometryPayload* __restrict__ meshpayloads,
    NerfPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer

) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements) {
		return;
	}
	NerfPayload& payload = payloads[idx];
	GeometryPayload& meshpayload = meshpayloads[payload.idx];

	
	if(payload.idx != 0)
		printf("payload.idx: %d, payload.t: %f\n", payload.idx, payload.t);
	
	frame_buffer[meshpayload.idx * uint32_t(10)] = {0.0f, 255.0f, 0.0f, 1.0f};

	depth_buffer[meshpayload.idx] = 1.0f;
}

__global__ void printArray(const uint32_t n_elements, cudaTextureObject_t texObj, const ivec2 numsampels) {
    uint32_t u = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t v = threadIdx.y + blockDim.y * blockIdx.y;
    if (u >= numsampels.x || v >= numsampels.y) {
        return;
    }

    // Fetch the value from the texture
    float4 value = tex2D<float4>(texObj, u, v);
	if(value.x > 0.0f || value.y > 0.0f || value.w > 0.0f)
    	printf("envmap: (%f, %f, %f, %f)\n", value.x, value.y, value.z, value.w);
}

__global__ void visualize_veiwpos(
    const uint32_t n_elements,
    NerfPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer

) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n_elements) {
		return;
	}
	NerfPayload& payload = payloads[idx];
	
	frame_buffer[payload.idx] = {payload.origin.x, 0.0f, 0.0f, 1.0f};

	depth_buffer[payload.idx] = 0.0f;
}

__global__ void printEnvMapArray(cudaArray_t envmapArray, const ivec2 gridSize, const ivec2 numSamples) {
    int idxPosition = threadIdx.x + blockDim.x * blockIdx.x;
    int idxDirection = threadIdx.y + blockDim.y * blockIdx.y;

    if (idxPosition >= gridSize.x * gridSize.y || idxDirection >= numSamples.x * numSamples.y) {
        return;
    }

    int layer = idxPosition;
    int u = idxDirection % numSamples.x;
    int v = idxDirection / numSamples.x;

    cudaArray_t layerArray;
    cudaExtent extent;
  
    float4 value;
    // cudaMemcpyFromArray(&value, layerArray, u * sizeof(float4), v, sizeof(float4), cudaMemcpyDeviceToHost);

    // printf("envmapArray: (%f, %f, %f, %f)\n", value.x, value.y, value.z, value.w);
}

__global__ void printNewEnvMap(const uint32_t n_elements, const ivec2 gridSize, const ivec2 numsampels, const cudaTextureObject_t newTexObj) {
    uint32_t thetaMul = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t phiMul = threadIdx.y + blockDim.y * blockIdx.y;
    uint32_t idxPosition = threadIdx.z + blockDim.z * blockIdx.z;

    if (idxPosition >= gridSize.x * gridSize.y || thetaMul >= numsampels.x || phiMul >= numsampels.y) {
        return;
    }

	float cos_theta = static_cast<float>(thetaMul)/numsampels.x;
	float phi = static_cast<float>(phiMul)/numsampels.y;
	auto layer = idxPosition;
	// printf("after calculation: cos_theta: %f, phi: %f, layer: %d\n", cos_theta, phi, layer);

    float4 value = tex2DLayered<float4>(newTexObj, cos_theta, phi, layer);
    
	// if(value.x > 0.0f || value.y > 0.0f || value.w > 0.0f)
        // printf("new envmap: (%f, %f, %f, %f)\n", value.x, value.y, value.z, value.w);
}

void printEnvmap(
	const ivec2 gridSize, 
	const ivec2 numsampels,
	cudaTextureObject_t *envmapTex,
	cudaStream_t stream
) {
	size_t n_rays = (size_t)numsampels.x * numsampels.y * gridSize.x * gridSize.y;

	const dim3 threads = { 16, 8, 1 };

	const dim3 blocks = { div_round_up((uint32_t)numsampels.x , threads.x), div_round_up((uint32_t)numsampels.y  , threads.y), div_round_up((uint32_t)gridSize.x * gridSize.y  , threads.z) };
	
	printNewEnvMap<<<blocks, threads, 0, stream>>>(n_rays, gridSize, numsampels, *envmapTex);
	
	cudaStreamSynchronize(stream);


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mesh
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Testbed::MyTracer::init_rays_from_camera_mesh(
	uint32_t sample_index,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Buffer2DView<const vec4>& envmap,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x * resolution.y;
	enlarge_mesh(n_pixels, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	init_rays_with_payload_kernel_mesh_geometry<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays_mesh[0].pos,
		m_rays_mesh[0].distance,
		m_rays_mesh[0].payload,
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		aabb,
		floor_y,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask
	);
	m_n_rays_initialized_mesh = (uint32_t)n_pixels;
	// tlog::info() << "m_n_rays_initialized_mesh: " << m_n_rays_initialized_mesh;
}

void Testbed::MyTracer::init_rays_from_data_mesh(uint32_t n_elements, const RaysMeshSoa& data, cudaStream_t stream) {
	enlarge_mesh(n_elements, stream);
	m_rays_mesh[0].copy_from_other_async(n_elements, data, stream);
	m_n_rays_initialized_mesh = n_elements;
}

void Testbed::MyTracer::init_rays_from_data_mesh_multiple(uint32_t n_elements, cudaStream_t stream) {
	size_t n_rays = (size_t) n_elements;

	enlarge_mesh(n_rays, stream);
	
	m_n_rays_initialized_mesh = n_elements;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_mesh[0].pos, 				0, n_elements * sizeof(vec3), 				stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_mesh[0].normal, 			0, n_elements * sizeof(vec3), 				stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_mesh[0].distance,			0, n_elements * sizeof(float), 				stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_mesh[0].prev_distance, 		0, n_elements * sizeof(float), 				stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_mesh[0].total_distance, 	0, n_elements * sizeof(float), 				stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_mesh[0].payload, 			0, n_elements * sizeof(GeometryPayload), 	stream));

}

// void Testbed::NerfTracer::init_rays_from_data_old_version(
// 	uint32_t n_elements, 
// 	uint32_t padded_output_width, 
// 	uint32_t n_extra_dims, 
// 	const ivec2& resolution, 
// 	const vec2& screen_center, 
// 	const vec3& parallax_shift, 
// 	bool snap_to_pixel_centers, 
// 	const BoundingBox& render_aabb, 
// 	const mat3& render_aabb_to_local,
// 	const uint32_t n_rays_per_sample, 
// 	const RaysMeshSoa& data, 
// 	vec4* frame_buffer, 
// 	float* depth_buffer, 
// 	default_rng_t rng,
// 	cudaStream_t stream
// ) {
// 	size_t n_rays = (size_t) n_elements * n_rays_per_sample;
// 	// size_t n_rays = (size_t) resolution.x * resolution.y  * n_rays_per_sample * n_rays_per_sample;

// 	// works fine, without pixelization
// 	enlarge(n_rays, padded_output_width, n_extra_dims, stream);
// 	// careshes out of memory, without pixelization
	
// 	const dim3 threads = { 16, 8, 1 };
// 	// const dim3 blocks = { div_round_up((uint32_t)n_rays , threads.x), div_round_up((uint32_t)n_rays , threads.y), 1 };
// 	// careshes out of memory, without pixelization
// 	const dim3 blocks = { div_round_up((uint32_t)resolution.x * n_rays_per_sample , threads.x), div_round_up((uint32_t)resolution.y * n_rays_per_sample , threads.y), 1 };
	
	
// 	GPUMemory<vec2> random_numbers(n_rays);
// 	// // careshes out of memory, without pixelization

//     generate_random_uniform<float>(stream, rng, n_rays * 2, (float*)random_numbers.data());
// 	// careshes out of memory, without pixelization
	
// 	init_rays_with_payload_kernel_shade_geometry<<<blocks, threads, 0, stream>>>(
// 		n_rays_per_sample,
// 		m_rays[0].payload,
// 		resolution.x * resolution.y,
// 		screen_center,
// 		parallax_shift,
// 		snap_to_pixel_centers,
// 		render_aabb,
// 		render_aabb_to_local,
// 		data.pos,
// 		data.distance,
// 		data.payload,
// 		random_numbers.data(),
// 		resolution
// 	);
// 	// works fine, with pixelization
// 	// return;
// 	m_n_rays_initialized = resolution.x * resolution.y * n_rays_per_sample * n_rays_per_sample;
// 	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
// 	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));
// }

void Testbed::NerfTracer::init_rays_from_data(
	uint32_t n_elements, 
	uint32_t padded_output_width, 
	uint32_t n_extra_dims, 
	cudaStream_t stream
) {
	size_t n_rays = (size_t) n_elements;

	enlarge(n_rays, padded_output_width, n_extra_dims, stream);
	
	m_n_rays_initialized = n_elements;
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].payload, 0, m_n_rays_initialized * sizeof(NerfPayload), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));
}

uint32_t Testbed::MyTracer::trace_mesh_bvh(GeometryBvh* bvh, const MeshData* meshes, cudaStream_t stream) {
	uint32_t n_alive = m_n_rays_initialized_mesh;
	m_n_rays_initialized_mesh = 0;

	if (!bvh) {
		return 0;
	}

	// Abuse the normal buffer to temporarily hold ray directions
	parallel_for_gpu(stream, n_alive, [payloads=m_rays_mesh[0].payload, normals=m_rays_mesh[0].normal] __device__ (size_t i) {
		normals[i] = payloads[i].dir;
	});

	bvh->ray_trace_mesh_gpu(n_alive, m_rays_mesh[0].pos, m_rays_mesh[0].normal, meshes, stream);
	return n_alive;
}

// allocate and distribute workspace memory for rays
void Testbed::MyTracer::enlarge_mesh(size_t n_elements, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays[0]
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays[1]
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays_hit

		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc_mesh,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays_mesh[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), std::get<6>(scratch));
	m_rays_mesh[1].set(std::get<7>(scratch), std::get<8>(scratch), std::get<9>(scratch), std::get<10>(scratch), std::get<11>(scratch), std::get<12>(scratch), std::get<13>(scratch));
	m_rays_hit_mesh.set(std::get<14>(scratch), std::get<15>(scratch), std::get<16>(scratch), std::get<17>(scratch), std::get<18>(scratch), std::get<19>(scratch), std::get<20>(scratch));

	m_hit_counter_mesh = std::get<21>(scratch);
	m_alive_counter_mesh = std::get<22>(scratch);
}

uint32_t Testbed::NerfTracer::shade_from_nerf(
	const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
	const BoundingBox& render_aabb,
	float min_transmittance,
	uint32_t n_elements,
	cudaStream_t stream)
{
	

	uint32_t n_alive = m_n_rays_initialized;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];	//should enlarge the rays_nerf
		RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		// {
		// 	CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
		// 	linear_kernel(compact_kernel_nerf, 0, stream,
		// 		n_alive,
		// 		rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
		// 		rays_current.rgba, rays_current.depth, rays_current.payload,
		// 		m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
		// 		m_alive_counter, m_hit_counter
		// 	);
		// 	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
		// 	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		// }

		if (n_alive == 0) {
			break;
		}

		// Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
		uint32_t target_n_queries = 2 * 1024 * 1024;
		uint32_t n_steps_between_compaction = clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

		uint32_t extra_stride = network->n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
		// linear_kernel(generate_next_nerf_network_inputs, 0, stream,
		// 	n_alive,
		// 	render_aabb,
		// 	render_aabb_to_local,
		// 	train_aabb,
		// 	focal_length,
		// 	camera_matrix[2],
		// 	rays_current.payload,
		// 	input_data,
		// 	n_steps_between_compaction,
		// 	grid,
		// 	(show_accel>=0) ? show_accel : 0,
		// 	max_mip,
		// 	cone_angle_constant,
		// 	extra_dims_gpu
		// );
		uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, BATCH_SIZE_GRANULARITY);
		GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
		GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network->padded_output_width(), n_elements);
		network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		// if (render_mode == ERenderMode::Normals) {
		// 	network->input_gradient(stream, 3, positions_matrix, positions_matrix);
		// } else if (render_mode == ERenderMode::EncodingVis) {
		// 	network->visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		// }

		// linear_kernel(composite_kernel_nerf, 0, stream,	//definitly should use this
		// 	n_alive,
		// 	n_elements,
		// 	i,
		// 	train_aabb,
		// 	glow_y_cutoff,
		// 	glow_mode,
		// 	camera_matrix,
		// 	focal_length,
		// 	depth_scale,
		// 	rays_current.rgba,
		// 	rays_current.depth,
		// 	rays_current.payload,
		// 	input_data,
		// 	m_network_output,
		// 	network->padded_output_width(),
		// 	n_steps_between_compaction,
		// 	render_mode,
		// 	grid,
		// 	rgb_activation,
		// 	density_activation,
		// 	show_accel,
		// 	min_transmittance
		// );

		i += n_steps_between_compaction;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::computeEnvmap(
    ivec2 numSamples,
	cudaTextureObject_t *envmapTex,
    const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network, 
	const BoundingBox& nerfBoundingBox,
	const BoundingBox& render_aabb,
	const mat3 render_aabb_to_local,
    float focal_length, 
    ERenderMode render_mode, 
    mat4 camera_matrix1, 
    uint32_t m_visualized_layer, 
    uint32_t visualized_dimension, 
	const uint8_t* density_grid_bitfield,
    cudaStream_t stream
) {

	// I am not sure where this should be computed, maybe even after loading the nerf
	// calculate the precomputed envmap
	// 1. initialize the rays from the nerfaabb center towards all thetas and phis
	// 2. trace the rays
	// 3. store the rgbs and dir in m_light_envmap



    NerfTracer envmap_tracer;

    envmap_tracer.init_rays_from_center(
        numSamples.x, 
		numSamples.y,
        nerf_network->padded_output_width(),
        nerf_network->n_extra_dims(),
        nerfBoundingBox,
        render_aabb_to_local,
        stream
    );

	float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;
	const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);
		

    uint32_t n_hit_envmap = envmap_tracer.trace_mesh(
		nerf_network,
		render_aabb,
		render_aabb_to_local,
		nerfBoundingBox,
		focal_length,
		m_geometry.nerf.cone_angle_constant,
		density_grid_bitfield,
		render_mode,
		camera_matrix1,
		depth_scale,
		m_visualized_layer,
		visualized_dimension,
		m_geometry.nerf.rgb_activation,
		m_geometry.nerf.density_activation,
		m_geometry.nerf.show_accel,
		m_geometry.nerf.max_cascade,
		m_geometry.nerf.render_min_transmittance,
		m_geometry.nerf.glow_y_cutoff,
		m_geometry.nerf.glow_mode,
		extra_dims_gpu,
		stream
	);
    

    auto& envmap_rays_hit = envmap_tracer.rays_hit();
	auto numsamplesmul = numSamples.x * numSamples.y;
    
	GPUMemory<vec4> envmap_rgb(numSamples.x * numSamples.y);

    linear_kernel(write_envmap_ray_result_from_nerf, 0, stream,
            n_hit_envmap,
            envmap_rays_hit.rgba,
            envmap_rays_hit.payload,
            envmap_rgb.data()
    );

	cudaArray* envmapArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK_THROW(cudaMallocArray(&envmapArray, &channelDesc, numSamples.x, numSamples.y));

    CUDA_CHECK_THROW(cudaMemcpyToArray(envmapArray, 0, 0, envmap_rgb.data(), numsamplesmul * sizeof(vec4), cudaMemcpyDeviceToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envmapArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

	//  theta should be clamped
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
	texDesc.sRGB = 0;
	texDesc.borderColor[0] = 0.0f;
    texDesc.borderColor[1] = 0.0f;
    texDesc.borderColor[2] = 0.0f;
    texDesc.borderColor[3] = 0.0f;
    texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.mipmapLevelBias = 0.0f;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = 0.0f;
    texDesc.disableTrilinearOptimization = 0;
    texDesc.seamlessCubemap = 0;

    CUDA_CHECK_THROW(cudaCreateTextureObject(envmapTex, &resDesc, &texDesc, nullptr));

    CUDA_CHECK_THROW(cudaDeviceSynchronize());

	// cudaResourceDesc resDesc;
	// memset(&resDesc, 0, sizeof(resDesc));
	// resDesc.resType = cudaResourceTypePitch2D;
	// resDesc.res.pitch2D.devPtr = envmap_rgb.data();
	// resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	// resDesc.res.pitch2D.width = numSamples.x;
	// resDesc.res.pitch2D.height = numSamples.y;
	// resDesc.res.pitch2D.pitchInBytes = numSamples.x * 4 * sizeof(vec3);

	// *envmapTex = texture;

	// cudaTextureDesc texDesc;
	// memset(&texDesc, 0, sizeof(texDesc));
	// texDesc.filterMode = cudaFilterModeLinear;
	// texDesc.normalizedCoords = true;
	// texDesc.addressMode[0] = cudaAddressModeClamp;
	// texDesc.addressMode[1] = cudaAddressModeClamp;
	// // texDesc.addressMode[2] = cudaAddressModeClamp;

	// cudaTextureObject_t texture;

	// CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));


    // cudaResourceDesc resDesc;
    // memset(&resDesc, 0, sizeof(resDesc));
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = envmapArray;


    // linear_kernel(printArray, 0, stream, 
    // 	n_hit_envmap, 
    // 	*envmapTex, 
    // 	numSamples);

}

void Testbed::computeEnvmapMultiple(
    ivec2 numSamples,
	uint32_t numSamplesOrigin,
	cudaTextureObject_t *envmapTex,
    const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network, 
	const BoundingBox& nerfBoundingBox,
	const BoundingBox& render_aabb,
	const mat3 render_aabb_to_local,
    float focal_length, 
    ERenderMode render_mode, 
    mat4 camera_matrix1, 
    uint32_t m_visualized_layer, 
    uint32_t visualized_dimension, 
	const uint8_t* density_grid_bitfield,
    cudaStream_t stream
) {

	// 1. initialize the rays from the nerfaabb center towards all thetas and phis
	// 2. trace the rays
	// 3. store the rgbs and dir in m_light_envmap

    NerfTracer envmap_tracer;

	// put this in a for loop over all the samples * number
    envmap_tracer.init_rays_from_multiple_center(
        numSamples.x, 
		numSamples.y,
		numSamplesOrigin,
        nerf_network->padded_output_width(),
        nerf_network->n_extra_dims(),
        nerfBoundingBox,
        render_aabb_to_local,
        stream
    );

	float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;
	const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);
		
    uint32_t n_hit_envmap = envmap_tracer.trace_mesh(
		nerf_network,
		render_aabb,
		render_aabb_to_local,
		nerfBoundingBox,
		focal_length,
		m_geometry.nerf.cone_angle_constant,
		density_grid_bitfield,
		render_mode,
		camera_matrix1,
		depth_scale,
		m_visualized_layer,
		visualized_dimension,
		m_geometry.nerf.rgb_activation,
		m_geometry.nerf.density_activation,
		m_geometry.nerf.show_accel,
		m_geometry.nerf.max_cascade,
		m_geometry.nerf.render_min_transmittance,
		m_geometry.nerf.glow_y_cutoff,
		m_geometry.nerf.glow_mode,
		extra_dims_gpu,
		stream
	);
    

    auto& envmap_rays_hit = envmap_tracer.rays_hit();
	auto numsamplesmul = numSamples.x * numSamples.y;
    
	GPUMemory<vec4> envmap_rgb(numSamples.x * numSamples.y);

    linear_kernel(write_envmap_multiple_ray_result_from_nerf, 0, stream,
            n_hit_envmap,
			numSamplesOrigin,
            envmap_rays_hit.rgba,
            envmap_rays_hit.payload,
            envmap_rgb.data()
    );
	// till here
	
	
	
	cudaArray* envmapArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK_THROW(cudaMallocArray(&envmapArray, &channelDesc, numSamples.x, numSamples.y));

    CUDA_CHECK_THROW(cudaMemcpyToArray(envmapArray, 0, 0, envmap_rgb.data(), numsamplesmul * sizeof(vec4), cudaMemcpyDeviceToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envmapArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

	//  theta should be clamped
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
	texDesc.sRGB = 0;
	texDesc.borderColor[0] = 0.0f;
    texDesc.borderColor[1] = 0.0f;
    texDesc.borderColor[2] = 0.0f;
    texDesc.borderColor[3] = 0.0f;
    texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.mipmapLevelBias = 0.0f;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = 0.0f;
    texDesc.disableTrilinearOptimization = 0;
    texDesc.seamlessCubemap = 0;

    CUDA_CHECK_THROW(cudaCreateTextureObject(envmapTex, &resDesc, &texDesc, nullptr));

    CUDA_CHECK_THROW(cudaDeviceSynchronize());


}


void Testbed::computeEnvmapMultipleMain(

) {
	ivec2 numSamples(256, 256);
	uint32_t numSamplesOrigin(16);

	if (frobenius_norm(m_smoothed_camera - m_camera) < 0.001f) {
		m_smoothed_camera = m_camera;
	} else if (!m_camera_path.rendering) {
		reset_accumulation(true);
	}

	// Get the device currently used for rendering
	CudaDevice& device = primary_device();
	mat4x3 camera_matrix1 = m_smoothed_camera;
	vec2 orig_screen_center = m_screen_center;
	vec2 relative_focal_length = m_relative_focal_length;
	vec4 nerf_rolling_shutter = {0.0f, 0.0f, 0.0f, 1.0f};
	Foveation foveation = {};
	int visualized_dimension = 0;

	cudaTextureObject_t *envmapTex = &m_geometry.m_envmap_tex;

	std::shared_ptr<NerfNetwork<network_precision_t>> nerf_network  = device.geometry_nerf_network();
    
	BoundingBox nerfBoundingBox = m_geometry.nerfBoundingBox;
	BoundingBox	render_aabb = m_render_aabb;
	mat3 render_aabb_to_local = m_render_aabb_to_local;
	
	vec2 focal_length = calc_focal_length(device.render_buffer_view().resolution, relative_focal_length, m_fov_axis, m_zoom);
	
	uint8_t* density_grid_bitfield = device.data().density_grid_bitfield_ptr;
	
	cudaStream_t stream = device.stream();
	
	////////////////////////////////////////////////////////////////////////
	// 1. initialize the rays from the nerfaabb center towards all thetas and phis
	// 2. trace the rays
	// 3. store the rgbs and dir in m_light_envmap
	NerfTracer envmap_tracer;

	// put this in a for loop over all the samples * number
    envmap_tracer.init_rays_from_multiple_center(
        numSamples.x, 
		numSamples.y,
		numSamplesOrigin,
        nerf_network->padded_output_width(),
        nerf_network->n_extra_dims(),
        nerfBoundingBox,
        render_aabb_to_local,
        stream
    );

	float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;
	const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);
		
    uint32_t n_hit_envmap = envmap_tracer.trace_mesh(
		nerf_network,
		render_aabb,
		render_aabb_to_local,
		nerfBoundingBox,
		focal_length,
		m_geometry.nerf.cone_angle_constant,
		density_grid_bitfield,
		ERenderMode::ShadeEnvMap,
		camera_matrix1,
		depth_scale,
		m_visualized_layer,
		visualized_dimension,
		m_geometry.nerf.rgb_activation,
		m_geometry.nerf.density_activation,
		m_geometry.nerf.show_accel,
		m_geometry.nerf.max_cascade,
		m_geometry.nerf.render_min_transmittance,
		m_geometry.nerf.glow_y_cutoff,
		m_geometry.nerf.glow_mode,
		extra_dims_gpu,
		stream
	);
    

    auto& envmap_rays_hit = envmap_tracer.rays_hit();
	auto numsamplesmul = numSamples.x * numSamples.y;
    
	GPUMemory<vec4> envmap_rgb(numSamples.x * numSamples.y);

    linear_kernel(write_envmap_multiple_ray_result_from_nerf, 0, stream,
            n_hit_envmap,
			numSamplesOrigin,
            envmap_rays_hit.rgba,
            envmap_rays_hit.payload,
            envmap_rgb.data()
    );
	
	cudaArray* envmapArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK_THROW(cudaMallocArray(&envmapArray, &channelDesc, numSamples.x, numSamples.y));

    CUDA_CHECK_THROW(cudaMemcpyToArray(envmapArray, 0, 0, envmap_rgb.data(), numsamplesmul * sizeof(vec4), cudaMemcpyDeviceToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envmapArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

	//  theta should be clamped
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;

	// ca be changed to linear for smoother results
    texDesc.filterMode = cudaFilterModePoint;

    texDesc.readMode = cudaReadModeElementType;
	texDesc.sRGB = 0;
	texDesc.borderColor[0] = 0.0f;
    texDesc.borderColor[1] = 0.0f;
    texDesc.borderColor[2] = 0.0f;
    texDesc.borderColor[3] = 0.0f;
    texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.mipmapLevelBias = 0.0f;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = 0.0f;
    texDesc.disableTrilinearOptimization = 0;
    texDesc.seamlessCubemap = 0;

    CUDA_CHECK_THROW(cudaCreateTextureObject(envmapTex, &resDesc, &texDesc, nullptr));

    CUDA_CHECK_THROW(cudaDeviceSynchronize());

}

void Testbed::computeEnvmapGrid(

) {

	ivec2 numSamples(16, 16);

	if (frobenius_norm(m_smoothed_camera - m_camera) < 0.001f) {
		m_smoothed_camera = m_camera;
	} else if (!m_camera_path.rendering) {
		reset_accumulation(true);
	}

	// Get the device currently used for rendering
	CudaDevice& device = primary_device();
	mat4x3 camera_matrix1 = m_smoothed_camera;
	vec2 orig_screen_center = m_screen_center;
	vec2 relative_focal_length = m_relative_focal_length;
	vec4 nerf_rolling_shutter = {0.0f, 0.0f, 0.0f, 1.0f};
	Foveation foveation = {};
	int visualized_dimension = 0;


	std::shared_ptr<NerfNetwork<network_precision_t>> nerf_network  = primary_device().geometry_nerf_network();
    
	BoundingBox nerfBoundingBox = m_geometry.nerfBoundingBox;
	BoundingBox	render_aabb = m_render_aabb;
	mat3 render_aabb_to_local = m_render_aabb_to_local;
	
	vec2 focal_length = calc_focal_length(device.render_buffer_view().resolution, relative_focal_length, m_fov_axis, m_zoom);
	
	uint8_t* density_grid_bitfield = device.data().density_grid_bitfield_ptr;
	
	cudaStream_t stream = device.stream();
	cudaTextureObject_t *envmapTex = &m_geometry.m_envmap_tex;
	m_geometry.gridSize = ivec2(16, 16); 
	
	////////////////////////////////////////////////////////////////////////
	/*
	1. divide the bounding sphere into a grid of cells with theta and phis
	2. for each cell: 
			calculate the center of the cell
			initialize the rays from the cell center towards all thetas and phis
	 		trace the rays
			store the rgbs and dir in envmap in the cell idx
	*/
	
	// ivec2 gridSize(16, 16);
	int totalCells = m_geometry.gridSize.x * m_geometry.gridSize.y;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	
	
	cudaArray* envmapArray;
	CUDA_CHECK_THROW(cudaMalloc3DArray(&envmapArray, &channelDesc, make_cudaExtent(numSamples.x, numSamples.y, totalCells), cudaArrayLayered));
	
	NerfTracer envmap_tracer;

	// x : theta [0, pi]
	// y : phi [0, 2pi]
	// z : index of the cell

    for (int positionThetaIndex = 0; positionThetaIndex < m_geometry.gridSize.x; ++positionThetaIndex) {
		for (int positionPhiIndex = 0; positionPhiIndex < m_geometry.gridSize.y; ++positionPhiIndex) {
			
			int i = positionThetaIndex + m_geometry.gridSize.x * positionPhiIndex;
			// printf("inside calc i = %d\n", i);
        	// Calculate the theta and phi of the cell
			// 0 < thetaMul/numSamplesTheta < 1
			// 0 < phiMul/numSamplesPhi < 1

			// -1 < cos_theta < 1
			// -PI < phi < PI

        	float cos_theta = static_cast<float>(positionThetaIndex) / m_geometry.gridSize.x;
        	float phi = static_cast<float>(positionPhiIndex) / m_geometry.gridSize.y;

        	vec3 dir = cylindrical_to_dir(vec2(cos_theta, phi));

        	// Calculate the center of the cell in Cartesian coordinates
        	vec3 cellCenter = nerfBoundingBox.center() + nerfBoundingBox.radius() * dir;
    	    
			// Initialize rays from the center of the cell towards all thetas and phis inwards with changing of the coordinates
			envmap_tracer.init_rays_from_center_outward(
    	        numSamples.x, 
    	        numSamples.y,
    	        nerf_network->padded_output_width(),
    	        nerf_network->n_extra_dims(),
    	        nerfBoundingBox,
    	        render_aabb_to_local,
				cellCenter,
    	        stream
    	    );
			
			float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;
			const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);
    		
			uint32_t n_hit_envmap = envmap_tracer.trace_mesh(
				nerf_network,
				render_aabb,
				render_aabb_to_local,
				nerfBoundingBox,
				focal_length,
				m_geometry.nerf.cone_angle_constant,
				density_grid_bitfield,
				ERenderMode::ShadeEnvMap,
				camera_matrix1,
				depth_scale,
				m_visualized_layer,
				visualized_dimension,
				m_geometry.nerf.rgb_activation,
				m_geometry.nerf.density_activation,
				m_geometry.nerf.show_accel,
				m_geometry.nerf.max_cascade,
				m_geometry.nerf.render_min_transmittance,
				m_geometry.nerf.glow_y_cutoff,
				m_geometry.nerf.glow_mode,
				extra_dims_gpu,
				stream
			);
    		auto& envmap_rays_hit = envmap_tracer.rays_hit();
			auto numsamplesmul = numSamples.x * numSamples.y;
			GPUMemory<vec4> envmap_rgb(numSamples.x * numSamples.y);
    		linear_kernel(write_envmap_ray_result_from_nerf, 0, stream,
    		        n_hit_envmap,
    		        envmap_rays_hit.rgba,
    		        envmap_rays_hit.payload,
    		        envmap_rgb.data()
    		);

        	cudaMemcpy3DParms copyParams = {0};
        	copyParams.srcPtr = make_cudaPitchedPtr(envmap_rgb.data(), numSamples.x * sizeof(float4), numSamples.x, numSamples.y); 
        	copyParams.dstArray = envmapArray;
        	copyParams.dstPos = make_cudaPos(0, 0, i);
        	copyParams.extent = make_cudaExtent(numSamples.x, numSamples.y, 1);
        	copyParams.kind = cudaMemcpyDeviceToDevice;
        	CUDA_CHECK_THROW(cudaMemcpy3D(&copyParams));
			CUDA_CHECK_THROW(cudaDeviceSynchronize());
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// vec4* host_envmap_rgb = new vec4[numSamples.x * numSamples.y];

			// // Copy the data from the GPU to the CPU
			// CUDA_CHECK_THROW(cudaMemcpy(host_envmap_rgb, envmap_rgb.data(), numSamples.x * numSamples.y * sizeof(vec4), cudaMemcpyDeviceToHost));

			// // Print the data
			// for (int j = 0; j < numSamples.x * numSamples.y; ++j) {
			//     printf("envmap_rgb[%d] = (%f, %f, %f, %f)\n", j, host_envmap_rgb[j].x, host_envmap_rgb[j].y, host_envmap_rgb[j].z, host_envmap_rgb[j].w);
			// }

			// float4* hostArray = new float4[gridSize.x * gridSize.y * numSamples.x * numSamples.y];

			// // Set up the cudaMemcpy3D parameters
			// cudaMemcpy3DParms copyParamsprint = {0};
			// copyParamsprint.srcArray = envmapArray;
			// copyParamsprint.dstPtr = make_cudaPitchedPtr(hostArray, numSamples.x * sizeof(float4), numSamples.x, numSamples.y);
			// copyParamsprint.extent = make_cudaExtent(numSamples.x, numSamples.y, gridSize.x * gridSize.y);
			// copyParamsprint.kind = cudaMemcpyDeviceToHost;

			// // Copy data from device to host
			// CUDA_CHECK_THROW(cudaMemcpy3D(&copyParamsprint));

			// // Print the host array
			// for (int i = 0; i < gridSize.x * gridSize.y * numSamples.x * numSamples.y; ++i) {
			//     printf("hostArray: (%f, %f, %f, %f)\n", hostArray[i].x, hostArray[i].y, hostArray[i].z, hostArray[i].w);
			// }

			// delete[] hostArray;
    	}
    }

	// Create a texture object for the 3D array
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = envmapArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));

	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeClamp;

	// cudaFilterModePoint = nearest-neighbor sampling method
	// cudaFilterModeLinear = linear interpolation
	texDesc.filterMode = cudaFilterModePoint;

	texDesc.readMode = cudaReadModeElementType;
	
	texDesc.sRGB = 0;
	texDesc.borderColor[0] = 0.0f;
    texDesc.borderColor[1] = 0.0f;
    texDesc.borderColor[2] = 0.0f;
    texDesc.borderColor[3] = 0.0f;
    texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.mipmapLevelBias = 0.0f;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = 0.0f;
    texDesc.disableTrilinearOptimization = 0;
    texDesc.seamlessCubemap = 0;

	CUDA_CHECK_THROW(cudaCreateTextureObject(envmapTex, &resDesc, &texDesc, nullptr));
	
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	// linear_kernel(printNewEnvMap, 0, stream, 
    // 	gridSize.x * gridSize.y * numSamples.x * numSamples.y, 
    // 	*envmapTex,
	// 	gridSize, 
    // 	numSamples);

	// printEnvmap(m_geometry.gridSize, numSamples, envmapTex, stream);
}


void Testbed::render_geometry_mesh(
	cudaStream_t stream,
	const distance_fun_t& distance_function,
	const normals_fun_t& normals_function,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension,

	CudaDevice& device,
	const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
	const uint8_t* density_grid_bitfield,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter
) {
	
	float plane_z = m_slice_plane_z + m_scale;

	MyTracer tracer;

	BoundingBox bounding_box = m_aabb;

	bounding_box.inflate(m_geometry.mesh_cpu[0].sdf.zero_offset);
	tracer.init_rays_from_camera_mesh(
		render_buffer.spp,
		render_buffer.resolution,
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		bounding_box,
		get_floor_y(),
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		m_envmap.inference_view(),
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		stream
	);
	
	auto trace = [&](MyTracer& tracer) {
		return tracer.trace_mesh_bvh(m_geometry.geometry_mesh_bvh.get(), m_geometry.mesh_cpu.data(), stream);
	};

	uint32_t n_hit = trace(tracer);

	RaysMeshSoa& rays_hit =  tracer.rays_init();


	ERenderMode render_mode = (visualized_dimension > -1 || m_render_mode == ERenderMode::Slice) ? ERenderMode::EncodingVis : m_render_mode;

		
	if (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Normals) {
		
		normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);

		if (render_mode == ERenderMode::Shade && n_hit > 0) {
			// Shadow rays towards the sun
			MyTracer shadow_tracer;

			// copy from the hit rays to the shadow rays
			shadow_tracer.init_rays_from_data_mesh(n_hit, rays_hit, stream);
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_geometry.mesh_cpu[0].sdf.shadow_sharpness);
			RaysMeshSoa& shadow_rays_init = shadow_tracer.rays_init();
			
			// changes the dir of the shadow rays to the sun dir and move the origin to the aabb 
			linear_kernel(prepare_shadow_rays_geometry, 0, stream,
				n_hit,
				normalize(m_sun_dir),
				shadow_rays_init.pos,
				shadow_rays_init.normal,
				shadow_rays_init.distance,
				shadow_rays_init.prev_distance,
				shadow_rays_init.total_distance,
				shadow_rays_init.min_visibility,
				shadow_rays_init.payload,
				bounding_box
			);

			uint32_t n_hit_shadow = trace(shadow_tracer);
			auto& shadow_rays_hit = shadow_tracer.rays_init();

			linear_kernel(write_shadow_ray_result_geometry, 0, stream,
				n_hit_shadow,
				bounding_box,
				shadow_rays_hit.pos,
				shadow_rays_hit.payload,
				shadow_rays_hit.min_visibility,
				rays_hit.distance
			);
		}

		linear_kernel(shade_kernel_mesh_geometry, 0, stream,
			n_hit,
			bounding_box,
			get_floor_y(),
			render_mode,
			m_geometry.mesh_cpu[0].sdf.brdf,
			normalize(m_sun_dir),
			normalize(m_up_dir),
			camera_matrix,
			rays_hit.pos,
			rays_hit.normal,
			rays_hit.distance,
			rays_hit.payload,
			render_buffer.frame_buffer,
			render_buffer.depth_buffer,
			nullptr,
			vec3(0.0f),
			vec2(0.0f),
			nullptr
		);
	} 

	else if (render_mode == ERenderMode::ShadeNerf) {
		
		normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);

		if (n_hit > 0) {
			// Shadow rays towards the nerf
			NerfTracer shadow_tracer;

			auto n_rays_per_sample = uint32_t(16);
			
			size_t n_rays = (size_t) n_hit * n_rays_per_sample;

			// allocate memory for the shadow rays
			shadow_tracer.init_rays_from_data(
				n_rays, 
				nerf_network->padded_output_width(),
				nerf_network->n_extra_dims(),
				stream);
			
			
			// generate the random directions for the shadow rays
			GPUMemory<vec2> random_numbers(n_rays);
			generate_random_uniform<float>(stream, m_rng, n_rays * 2, (float*)random_numbers.data());
			// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			
			GPUMemory<float> weights(n_hit);

			// shadow_rays_init = m_rays[0]
			RaysNerfSoa& shadow_rays_init = shadow_tracer.rays_init();
			
			linear_kernel(init_rays_with_payload_kernel_shade_geometry, 0, stream,
				n_rays,
				shadow_rays_init.payload,
				n_rays_per_sample,
				m_geometry.nerfBoundingBox,	
				mat3::identity(),
				rays_hit.pos,
				rays_hit.normal,
				rays_hit.distance,
				rays_hit.payload,
				random_numbers.data(),
				weights.data()
			);
			
			float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;
			const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);
			
			uint32_t n_hit_shadow = shadow_tracer.trace_mesh(
				nerf_network,
				m_render_aabb,
				m_render_aabb_to_local,
				m_geometry.nerfBoundingBox,
				focal_length,
				m_geometry.nerf.cone_angle_constant,
				density_grid_bitfield,
				render_mode,
				camera_matrix1,
				depth_scale,
				m_visualized_layer,
				visualized_dimension,
				m_geometry.nerf.rgb_activation,
				m_geometry.nerf.density_activation,
				m_geometry.nerf.show_accel,
				m_geometry.nerf.max_cascade,
				m_geometry.nerf.render_min_transmittance,
				m_geometry.nerf.glow_y_cutoff,
				m_geometry.nerf.glow_mode,
				extra_dims_gpu,
				stream
			);
			
			auto& shadow_rays_hit = shadow_tracer.rays_hit();
				
			GPUMemory<vec3> rgb(n_hit);
			linear_kernel( setVec3, 0, stream, n_hit, rgb.data(), 0.0f);
			
			linear_kernel(write_shadow_ray_result_from_nerf, 0, stream,
					n_hit_shadow,
					shadow_rays_hit.rgba,
					shadow_rays_hit.payload,
					n_rays_per_sample,
					rgb.data(),
					weights.data()
				);

			CUDA_CHECK_THROW(cudaDeviceSynchronize());

			linear_kernel(shade_kernel_from_nerf, 0, stream,
				n_hit,
				bounding_box,
				get_floor_y(),
				render_mode,
				m_geometry.mesh_cpu[0].sdf.brdf,
				normalize(m_up_dir),
				camera_matrix,
				rays_hit.pos,
				rays_hit.normal,
				rays_hit.distance,
				rays_hit.payload,
				render_buffer.frame_buffer,
				render_buffer.depth_buffer,
				m_geometry.nerfBoundingBox,
				rgb.data(),
				shadow_rays_hit.payload,
				n_rays_per_sample
			);

		}
	} 
	
	else if (render_mode == ERenderMode::ShadeEnvMap) {
		
		normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);
		
		ivec2 numSamples(128, 128);

		cudaTextureObject_t envmapTex = 0;
		
		// computeEnvmap(
		// 	numSamples,
		// 	&envmapTex,
		// 	nerf_network, 
		// 	m_geometry.nerfBoundingBox,
		// 	m_render_aabb,
		// 	m_render_aabb_to_local,
		// 	focal_length.x, 
		// 	render_mode, 
		// 	camera_matrix1, 
		// 	m_visualized_layer, 
		// 	visualized_dimension, 
		// 	density_grid_bitfield,
		// 	stream
		// );
		// computeEnvmapMultiple(
		// 	numSamples,
		// 	uint32_t(4),
		// 	&envmapTex,
		// 	nerf_network, 
		// 	m_geometry.nerfBoundingBox,
		// 	m_render_aabb,
		// 	m_render_aabb_to_local,
		// 	focal_length.x, 
		// 	render_mode, 
		// 	camera_matrix1, 
		// 	m_visualized_layer, 
		// 	visualized_dimension, 
		// 	density_grid_bitfield,
		// 	stream
		// );
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		// printf("Texture object: %llu\n", envmapTex);

		if (n_hit > 0) {
			// Shadow rays towards the center of the nerf
			MyTracer shadow_tracer;

			// copy from the hit rays to the shadow rays
			shadow_tracer.init_rays_from_data_mesh(n_hit, rays_hit, stream);
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_geometry.mesh_cpu[0].sdf.shadow_sharpness);
			RaysMeshSoa& shadow_rays_init = shadow_tracer.rays_init();
			
			linear_kernel(prepare_shadow_rays_envmap_geometry, 0, stream,
				n_hit,
				m_geometry.nerfBoundingBox.center(),
				shadow_rays_init.pos,
				shadow_rays_init.normal,
				shadow_rays_init.distance,
				shadow_rays_init.prev_distance,
				shadow_rays_init.total_distance,
				shadow_rays_init.min_visibility,
				shadow_rays_init.payload,
				bounding_box
			);

			uint32_t n_hit_shadow = trace(shadow_tracer);
			auto& shadow_rays_hit = shadow_tracer.rays_init();
			linear_kernel(write_shadow_ray_result_geometry, 0, stream,
				n_hit_shadow,
				bounding_box,
				shadow_rays_hit.pos,
				shadow_rays_hit.payload,
				shadow_rays_hit.min_visibility,
				rays_hit.distance
			);
			CUDA_CHECK_THROW(cudaDeviceSynchronize());
			linear_kernel(shade_kernel_mesh_geometry, 0, stream,
				n_hit,
				bounding_box,
				get_floor_y(),
				render_mode,
				m_geometry.mesh_cpu[0].sdf.brdf,
				normalize(m_sun_dir),
				normalize(m_up_dir),
				camera_matrix,
				rays_hit.pos,
				rays_hit.normal,
				rays_hit.distance,
				rays_hit.payload,
				render_buffer.frame_buffer,
				render_buffer.depth_buffer,
				&m_geometry.m_envmap_tex,
				m_geometry.nerfBoundingBox.center(),
				vec2(0.0f),
				nullptr
			);
			cudaDeviceSynchronize();
			cudaError err = cudaGetLastError();
			if (err != cudaSuccess) {
			    printf("CUDA error: %s\n", cudaGetErrorString(err));
			}

		}
	} 
	
	// I can change the rays from nerf rays to actual geometry rays 
	//  and just use a normal tracer to trace the rays
	// else if (render_mode == ERenderMode::ShadeGridEnvMap) {
		
	// 	normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);

	// 	if (n_hit > 0) {
			
	// 		NerfTracer shadow_tracer;

	// 		auto n_rays_per_sample = uint32_t(16);
			
	// 		size_t n_rays = (size_t) n_hit * n_rays_per_sample;

	// 		// allocate memory for the shadow rays
	// 		shadow_tracer.init_rays_from_data(
	// 			n_rays, 
	// 			nerf_network->padded_output_width(),
	// 			nerf_network->n_extra_dims(),
	// 			stream);
			
			
	// 		// generate the random directions for the shadow rays
	// 		GPUMemory<vec2> random_numbers(n_rays);
	// 		generate_random_uniform<float>(stream, m_rng, n_rays * 2, (float*)random_numbers.data());
	// 		// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			
	// 		GPUMemory<float> weights(n_hit);

	// 		// shadow_rays_init = m_rays[0]
	// 		RaysNerfSoa& shadow_rays_init = shadow_tracer.rays_init();
			
	// 		linear_kernel(init_rays_with_payload_kernel_shade_geometry, 0, stream,
	// 			n_rays,
	// 			shadow_rays_init.payload,
	// 			n_rays_per_sample,
	// 			m_geometry.nerfBoundingBox,	
	// 			mat3::identity(),
	// 			rays_hit.pos,
	// 			rays_hit.normal,
	// 			rays_hit.distance,
	// 			rays_hit.payload,
	// 			random_numbers.data(),
	// 			weights.data()
	// 		);
			
	// 		float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;
	// 		const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);
			
	// 		uint32_t n_hit_shadow = shadow_tracer.trace_mesh(
	// 			nerf_network,
	// 			m_render_aabb,
	// 			m_render_aabb_to_local,
	// 			m_geometry.nerfBoundingBox,
	// 			focal_length,
	// 			m_geometry.nerf.cone_angle_constant,
	// 			density_grid_bitfield,
	// 			render_mode,
	// 			camera_matrix1,
	// 			depth_scale,
	// 			m_visualized_layer,
	// 			visualized_dimension,
	// 			m_geometry.nerf.rgb_activation,
	// 			m_geometry.nerf.density_activation,
	// 			m_geometry.nerf.show_accel,
	// 			m_geometry.nerf.max_cascade,
	// 			m_geometry.nerf.render_min_transmittance,
	// 			m_geometry.nerf.glow_y_cutoff,
	// 			m_geometry.nerf.glow_mode,
	// 			extra_dims_gpu,
	// 			stream
	// 		);
			
	// 		auto& shadow_rays_hit = shadow_tracer.rays_hit();
				
	// 		GPUMemory<vec3> rgb(n_hit);
	// 		linear_kernel( setVec3, 0, stream, n_hit, rgb.data(), 0.0f);
			
	// 		linear_kernel(write_shadow_ray_result_from_nerf, 0, stream,
	// 				n_hit_shadow,
	// 				shadow_rays_hit.rgba,
	// 				shadow_rays_hit.payload,
	// 				n_rays_per_sample,
	// 				rgb.data(),
	// 				weights.data()
	// 			);

	// 		CUDA_CHECK_THROW(cudaDeviceSynchronize());

	// 		linear_kernel(shade_kernel_from_nerf, 0, stream,
	// 			n_hit,
	// 			bounding_box,
	// 			get_floor_y(),
	// 			render_mode,
	// 			m_geometry.mesh_cpu[0].sdf.brdf,
	// 			normalize(m_up_dir),
	// 			camera_matrix,
	// 			rays_hit.pos,
	// 			rays_hit.normal,
	// 			rays_hit.distance,
	// 			rays_hit.payload,
	// 			render_buffer.frame_buffer,
	// 			render_buffer.depth_buffer,
	// 			m_geometry.nerfBoundingBox,
	// 			rgb.data(),
	// 			shadow_rays_hit.payload,
	// 			n_rays_per_sample
	// 		);
	// 	}
	// } 

	else if (render_mode == ERenderMode::ShadeGridEnvMap) {
		
		normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);

		if (n_hit > 0) {
			// Shadow rays towards the scene
			MyTracer shadow_tracer;

			auto n_rays_per_sample = uint32_t(16);

			size_t n_rays = (size_t) n_hit * n_rays_per_sample;

			// allocate memory for the shadow rays
			shadow_tracer.init_rays_from_data_mesh_multiple(n_rays,  stream);
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_geometry.mesh_cpu[0].sdf.shadow_sharpness);
			
			// generate the random directions for the shadow rays
			GPUMemory<vec2> random_numbers(n_rays);
			generate_random_uniform<float>(stream, m_rng, n_rays * 2, (float*)random_numbers.data());
			// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			
			GPUMemory<float> weights(n_hit);

			RaysMeshSoa& shadow_rays_init = shadow_tracer.rays_init();
			
			linear_kernel(init_rays_with_payload_kernel_shade_raymesh, 0, stream,
				n_rays,
				shadow_rays_init.payload,
				shadow_rays_init.pos,
				shadow_rays_init.normal,
				n_rays_per_sample,
				m_geometry.nerfBoundingBox,	
				mat3::identity(),
				rays_hit.pos,
				rays_hit.normal,
				rays_hit.distance,
				rays_hit.payload,
				random_numbers.data(),
				weights.data()
			);

			// we do not have to trace the rays (?)
			// the position is updated to the intersection with the bounding box of nerf 
			// when do I check if a ray is alive or not
			uint32_t n_hit_shadow = n_rays;
			uint32_t* m_alive_counter;

			CUDA_CHECK_THROW(cudaMalloc((void**)&m_alive_counter, sizeof(uint32_t)));


			// Compact rays that did not diverge yet
			// maybe better in a function
			{
				CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
				linear_kernel(compact_kernel_shade, 0, stream,
					n_hit_shadow,
					shadow_rays_init.pos, shadow_rays_init.normal, shadow_rays_init.payload,
					shadow_tracer.rays_hit().pos, shadow_tracer.rays_hit().normal, shadow_tracer.rays_hit().payload,
					m_alive_counter
				);
				CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit_shadow, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			}

			auto& shadow_rays_hit = shadow_tracer.rays_hit();
			
			GPUMemory<vec3> rgb(n_hit);
			linear_kernel( setVec3, 0, stream, n_hit, rgb.data(), 0.0f);

			linear_kernel(write_shadow_ray_result_from_nerf_envmap_grid, 0, stream,
				n_hit_shadow,
				// bounding_box,
				shadow_rays_hit.pos,
				shadow_rays_hit.payload,
				shadow_rays_hit.min_visibility,
				n_rays_per_sample,
				rgb.data(),
				weights.data(),
				m_geometry.gridSize,
				&m_geometry.m_envmap_tex
			);

			linear_kernel(shade_kernel_mesh_geometry, 0, stream,
				n_hit,
				bounding_box,
				get_floor_y(),
				render_mode,
				m_geometry.mesh_cpu[0].sdf.brdf,
				normalize(m_sun_dir),
				normalize(m_up_dir),
				camera_matrix,
				rays_hit.pos,
				rays_hit.normal,
				rays_hit.distance,
				rays_hit.payload,
				render_buffer.frame_buffer,
				render_buffer.depth_buffer,
				nullptr,
				vec3(0.0f),
				vec2(0.0f),
				rgb.data()
			);
		}

	} 

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// nerf
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Testbed::render_geometry_nerf(
	cudaStream_t stream,
	CudaDevice& device,
	const CudaRenderBufferView& render_buffer,
	const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
	const uint8_t* density_grid_bitfield,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension
) {
	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	ERenderMode render_mode = visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

	const float* extra_dims_gpu = m_geometry.nerf.get_rendering_extra_dims(stream);

	NerfTracer tracer;

	// Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled.
	// (Unless we're in distortion visualization mode, in which case the distortion grid is fine to visualize.)
	auto grid_distortion =
		m_geometry.nerf.render_with_lens_distortion && (!m_dlss || m_render_mode == ERenderMode::Distortion) ?
		m_distortion.inference_view() :
		Buffer2DView<const vec2>{};

	Lens lens = m_geometry.nerf.render_with_lens_distortion ? m_geometry.nerf.render_lens : Lens{};

	auto resolution = render_buffer.resolution;

	tracer.init_rays_from_camera(
		render_buffer.spp,
		nerf_network->padded_output_width(),
		nerf_network->n_extra_dims(),
		render_buffer.resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		m_render_aabb,
		m_render_aabb_to_local,
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		lens,
		m_envmap.inference_view(),
		grid_distortion,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		density_grid_bitfield,
		m_geometry.nerf.show_accel,
		m_geometry.nerf.max_cascade,
		m_geometry.nerf.cone_angle_constant,
		render_mode,
		stream
	);

	float depth_scale = 1.0f / m_geometry.nerf.training.dataset.scale;

	uint32_t n_hit;
	
	n_hit = tracer.trace(
		nerf_network,
		m_render_aabb,
		m_render_aabb_to_local,
		m_geometry.nerfBoundingBox,
		focal_length,
		m_geometry.nerf.cone_angle_constant,
		density_grid_bitfield,
		render_mode,
		camera_matrix1,
		depth_scale,
		m_visualized_layer,
		visualized_dimension,
		m_geometry.nerf.rgb_activation,
		m_geometry.nerf.density_activation,
		m_geometry.nerf.show_accel,
		m_geometry.nerf.max_cascade,
		m_geometry.nerf.render_min_transmittance,
		m_geometry.nerf.glow_y_cutoff,
		m_geometry.nerf.glow_mode,
		extra_dims_gpu,
		stream
	);
	
	RaysNerfSoa& rays_hit = tracer.rays_hit();

	linear_kernel(shade_kernel_nerf_geometry, 0, stream,
		n_hit,
		m_geometry.nerf.render_gbuffer_hard_edges,
		camera_matrix1,
		depth_scale,
		rays_hit.rgba,
		rays_hit.depth,
		rays_hit.payload,
		m_render_mode,
		m_geometry.nerf.training.linear_colors,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer
	);

}


///////////////////////////////////////////////////////////////////////////////////////////////////
// loading functions
///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<vec3> geometry_load_stl(const fs::path& path) {
	std::vector<vec3> vertices;

	std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
	if (!f) {
		throw std::runtime_error{fmt::format("Mesh file '{}' not found", path.str())};
	}

	uint32_t buf[21] = {};
	f.read((char*)buf, 4 * 21);
	if (f.gcount() < 4 * 21) {
		throw std::runtime_error{fmt::format("Mesh file '{}' too small for STL header", path.str())};
	}

	uint32_t nfaces = buf[20];
	if (memcmp(buf, "solid", 5) == 0 || buf[20] == 0) {
		throw std::runtime_error{fmt::format("ASCII STL file '{}' not supported", path.str())};
	}

	vertices.reserve(nfaces * 3);
	for (uint32_t i = 0; i < nfaces; ++i) {
		f.read((char*)buf, 50);
		if (f.gcount() < 50) {
			nfaces = i;
			break;
		}

		vertices.push_back(*(vec3*)(buf + 3));
		vertices.push_back(*(vec3*)(buf + 6));
		vertices.push_back(*(vec3*)(buf + 9));
	}

	return vertices;
}

void Testbed::load_mesh(MeshData* mesh, const fs::path& data_path, vec3 center) {

	tlog::info() << "Loading mesh from '" << data_path << "'";
	auto start = std::chrono::steady_clock::now();

	std::vector<vec3> vertices;
	if (equals_case_insensitive(data_path.extension(), "obj")) {
		vertices = load_obj(data_path.str());
	} else if (equals_case_insensitive(data_path.extension(), "stl")) {
		vertices = geometry_load_stl(data_path.str());
	} else {
		throw std::runtime_error{"mesh data path must be a mesh in ascii .obj or binary .stl format."};
	}

	// The expected format is
	// [v1.x][v1.y][v1.z][v2.x]...
	size_t n_vertices = vertices.size();
	size_t n_triangles = n_vertices/3;


	// Compute the AABB of the mesh
	vec3 inf(std::numeric_limits<float>::infinity());
	BoundingBox aabb (inf, -inf);

	for (size_t i = 0; i < n_vertices; ++i) {
	    aabb.enlarge(vertices[i]);
	}

	// Inflate AABB by 1% to give the network a little wiggle room.
	const float inflation = 0.005f;

	aabb.inflate(length(aabb.diag()) * inflation);
	(*mesh).sdf.mesh_scale = max(aabb.diag());

	// Normalize the vertices.
	for (size_t i = 0; i < n_vertices; ++i) {
	    vertices[i] = (vertices[i] - aabb.min - 0.5f * aabb.diag()) / (*mesh).sdf.mesh_scale + 0.5f;
		vertices[i] += center;
	}

	BoundingBox aabb2 = {};	

	for (size_t i = 0; i < n_vertices; ++i) {
		aabb2.enlarge(vertices[i]);
	}

	aabb2.inflate(length(aabb2.diag()) * inflation);
	aabb2 = aabb2.intersection(BoundingBox{vec3(0.0f), vec3(1.0f)});

	(*mesh).center = center;
	

	// Normalize vertex coordinates to lie within [0,1]^3.
	// This way, none of the constants need to carry around
	// bounding box factors.
	// for (size_t i = 0; i < n_vertices; ++i) {
	// 	vertices[i] = (vertices[i] - aabb.min - 0.5f * aabb.diag()) / (*mesh).scale  + 0.5f;
	// }

	(*mesh).sdf.triangles_cpu.resize(n_triangles);
	for (size_t i = 0; i < n_vertices; i += 3) {
		(*mesh).sdf.triangles_cpu[i/3] = {vertices[i+0], vertices[i+1], vertices[i+2]};
	}

	if (!(*mesh).sdf.triangle_bvh) {
		(*mesh).sdf.triangle_bvh = TriangleBvh::make();
	}

	(*mesh).sdf.triangle_bvh->build((*mesh).sdf.triangles_cpu, 8);
	(*mesh).sdf.triangles_gpu.resize_and_copy_from_host((*mesh).sdf.triangles_cpu);

	
	(*mesh).sdf.triangle_weights.resize(n_triangles);
	for (size_t i = 0; i < n_triangles; ++i) {
		(*mesh).sdf.triangle_weights[i] = (*mesh).sdf.triangles_cpu[i].surface_area();
	}
	(*mesh).sdf.triangle_distribution.build((*mesh).sdf.triangle_weights);

	// Move CDF to gpu
	(*mesh).sdf.triangle_cdf.resize_and_copy_from_host((*mesh).sdf.triangle_distribution.cdf);

	tlog::success() << "Loaded mesh after " << tlog::durationToString(std::chrono::steady_clock::now() - start);
	tlog::info() << "  n_triangles=" << n_triangles;

}

void Testbed::load_empty_mesh(MeshData* mesh, vec3 center) {
	(*mesh).center = center;
	(*mesh).sdf.mesh_scale = 1.0f;
	load_mesh(mesh, "/home/fsalehi/test/instant-ngp/data/geometry/objs/simple.obj", center);

}

void Testbed::load_nerf_post(const vec3 center) { // moved the second half of load_nerf here
	m_geometry.nerf.rgb_activation = m_geometry.nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_geometry.nerf.training.n_images_for_training = (int)m_geometry.nerf.training.dataset.n_images;

	m_geometry.nerf.training.dataset.update_metadata();

	m_geometry.nerf.training.cam_pos_gradient.resize(m_geometry.nerf.training.dataset.n_images, vec3(0.0f));
	m_geometry.nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_geometry.nerf.training.cam_pos_gradient);

	m_geometry.nerf.training.cam_exposure.resize(m_geometry.nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
	m_geometry.nerf.training.cam_pos_offset.resize(m_geometry.nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
	m_geometry.nerf.training.cam_rot_offset.resize(m_geometry.nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	m_geometry.nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

	m_geometry.nerf.training.cam_rot_gradient.resize(m_geometry.nerf.training.dataset.n_images, vec3(0.0f));
	m_geometry.nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_geometry.nerf.training.cam_rot_gradient);

	m_geometry.nerf.training.cam_exposure_gradient.resize(m_geometry.nerf.training.dataset.n_images, vec3(0.0f));
	m_geometry.nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_geometry.nerf.training.cam_exposure_gradient);
	m_geometry.nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_geometry.nerf.training.cam_exposure_gradient);

	m_geometry.nerf.training.cam_focal_length_gradient = vec2(0.0f);
	m_geometry.nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_geometry.nerf.training.cam_focal_length_gradient, 1);

	m_geometry.nerf.reset_extra_dims(m_rng);
	m_geometry.nerf.training.optimize_extra_dims = m_geometry.nerf.training.dataset.n_extra_learnable_dims > 0;

	if (m_geometry.nerf.training.dataset.has_rays) {
		m_geometry.nerf.training.near_distance = 0.0f;
	}

	m_geometry.nerf.training.update_transforms();

	if (!m_geometry.nerf.training.dataset.metadata.empty()) {
		m_geometry.nerf.render_lens = m_geometry.nerf.training.dataset.metadata[0].lens;
		m_screen_center = vec2(1.f) - m_geometry.nerf.training.dataset.metadata[0].principal_point;
	}

	if (!is_pot(m_geometry.nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{fmt::format("m_geometry.nerf dataset's `aabb_scale` must be a power of two, but is {}.", m_geometry.nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (m_geometry.nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{fmt::format(
			"m_geometry.nerf dataset must have `aabb_scale <= {}`, but is {}. "
			"You can increase this limit by factors of 2 by incrementing `m_geometry.nerf_CASCADES()` and re-compiling.",
			max_aabb_scale, m_geometry.nerf.training.dataset.aabb_scale
		)};
	}

	m_geometry.nerfBoundingBox = BoundingBox{vec3(0.5f), vec3(0.5f)};
	m_geometry.nerfBoundingBox.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_geometry.nerf.training.dataset.aabb_scale));
	// m_raw_aabb = m_aabb;
	// m_render_aabb = m_aabb;
	m_render_aabb_to_local = m_geometry.nerf.training.dataset.render_aabb_to_local;
	if (!m_geometry.nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_geometry.nerf.training.dataset.render_aabb.intersection(m_aabb);
	}
	
	tlog::info() << "AABB: " << m_aabb;

	m_geometry.nerf.max_cascade = 0;
	while ((1 << m_geometry.nerf.max_cascade) < m_geometry.nerf.training.dataset.aabb_scale) {
		++m_geometry.nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_geometry.nerf.cone_angle_constant = m_geometry.nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = m_geometry.nerf.training.dataset.up;

	m_geometry.nerf.center = center;
	m_geometry.nerf.scale = m_geometry.nerf.training.dataset.aabb_scale;
}

void Testbed::load_nerf(const fs::path& data_path, const vec3 center) {
	if (!data_path.empty()) {
		std::vector<fs::path> json_paths;
		if (data_path.is_directory()) {
			for (const auto& path : fs::directory{data_path}) {
				if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
					json_paths.emplace_back(path);
				}
			}
		} else if (equals_case_insensitive(data_path.extension(), "json")) {
			json_paths.emplace_back(data_path);
		} else {
			throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
		}

		const auto prev_aabb_scale = m_geometry.nerf.training.dataset.aabb_scale;

		m_geometry.nerf.training.dataset = ngp::load_nerf(json_paths, m_geometry.nerf.sharpen);

		if (m_geometry.nerf.training.dataset.aabb_scale != prev_aabb_scale && m_geometry_nerf_network) {
			// The AABB scale affects network size indirectly. If it changed after loading,
			// we need to reset the previously configured network to keep a consistent internal state.
			reset_network();
		}

	}

	load_nerf_post(center);
}

void Testbed::load_scene(const fs::path& data_path) {


	/**
	 * [
    {
        "center": [0.0, 0.0, 0.0],
        "path": "path/to/geometry.obj",
        "type": "Mesh"
    },
    {
        "center": [1.0, 1.0, 1.0],
        "path": "path/to/geometry.json",
        "type": "Nerf"
    }
    // ... more geometries ...
	 *]
	 * 
	*/

 	size_t mesh_count = 0;
    size_t nerf_count = 0;


	if (!data_path.empty()) {
		if (m_geometry.geometry_mesh_bvh) {
			m_geometry.geometry_mesh_bvh.reset();
		}
		if (m_geometry.geometry_nerf_bvh) {
			m_geometry.geometry_nerf_bvh.reset();
		}
		std::ifstream f{native_string(data_path)};
		nlohmann::json jsonfile = nlohmann::json::parse(f, nullptr, true, true);

        if (jsonfile.empty()) {
            throw std::runtime_error{"Geometry file must contain an array of geometry metadata."};
        }

		nlohmann::json geometries = jsonfile["geometry"];

        // Count the number of Mesh and Nerf types
        for(auto& geometry : geometries) {
            std::string type = geometry["type"];
            if (type == "Mesh") {
                ++mesh_count;
            } else if (type == "Nerf") {
                ++nerf_count;
            }
        }

		std::cout << "Mesh count: " << mesh_count << std::endl;
		std::cout << "Nerf count: " << nerf_count << std::endl;

        // Resize the vectors
        m_geometry.mesh_cpu.resize(mesh_count);
        // m_geometry.nerf_cpu.resize(nerf_count);

        size_t mesh_index = 0;
        size_t nerf_index = 0;

        // Load the geometries
        for(auto& geometry : geometries) {
            fs::path model_path = geometry["path"];
            std::string type = geometry["type"];
            std::vector<float> center = geometry["center"];
            vec3 center_vec(center[0], center[1], center[2]);
			
            if (type == "Mesh") {
				// Todo: move constructor and delete copy constructor
                load_mesh(&m_geometry.mesh_cpu[mesh_index++],model_path, center_vec);
            } else if (type == "Nerf") {
				load_snapshot(model_path);
			} else {
				throw std::runtime_error{"Geometry type must be either 'Mesh' or 'Nerf'."};
            }
        }
	}
	
	if(mesh_count > 0) {
		m_geometry.geometry_mesh_bvh = GeometryBvh::make();
		// at the end we want each leaf to contain only one geometry
		tlog::info() << "Building mesh bvh";
		m_geometry.geometry_mesh_bvh->build_mesh(m_geometry.mesh_cpu, 1);
		const auto& nodes = m_geometry.geometry_mesh_bvh->get_nodes();
		BoundingBox bb = nodes[0].bb;
		bb.inflate(4.0);
		m_render_aabb = bb;
		m_render_aabb_to_local = mat3::identity();
		m_aabb = bb;

		// for (size_t i = 0; i < 5; ++i) {
		//     const auto& node = nodes[i];
		//     tlog::info() << "Node idx: " << i 
		//                  << ", bb: (" << node.bb.min.x << ", " << node.bb.min.y << ", " << node.bb.min.z 
		//                  << ", " << node.bb.max.x << ", " << node.bb.max.y << ", " << node.bb.max.z << ")"
		//                  << ", left_idx: " << node.left_idx 
		//                  << ", right_idx: " << node.right_idx;
		// }
		std::vector<const TriangleBvhNode*> bvhnodes(m_geometry.mesh_cpu.size());
    	std::vector<const Triangle*> triangles(m_geometry.mesh_cpu.size());

    	for (size_t i = 0; i < m_geometry.mesh_cpu.size(); ++i) {
    	    bvhnodes[i] = m_geometry.mesh_cpu[i].sdf.triangle_bvh->nodes_gpu();
    	    triangles[i] = m_geometry.mesh_cpu[i].sdf.triangles_gpu.data();
    	}

    	m_geometry.geometry_mesh_bvh->setBvhNodes(bvhnodes);
    	m_geometry.geometry_mesh_bvh->setTriangles(triangles);

		m_geometry.geometry_mesh_bvh->m_bvhnodes_gpu.resize_and_copy_from_host(m_geometry.geometry_mesh_bvh->getBvhNodes());
		m_geometry.geometry_mesh_bvh->m_triangles_gpu.resize_and_copy_from_host(m_geometry.geometry_mesh_bvh->getTriangles());
	}
	
	// ivec2 numSamples(128, 128);
	// computeEnvmap(
	// 		numSamples,
	// 		&m_geometry.m_envmap_tex,
	// 		nerf_network, 
	// 		render_buffer,
	// 		m_geometry.nerfBoundingBox,
	// 		m_render_aabb,
	// 		m_render_aabb_to_local,
	// 		focal_length.x, 
	// 		render_mode, 
	// 		camera_matrix1, 
	// 		m_visualized_layer, 
	// 		visualized_dimension, 
	// 		density_grid_bitfield,
	// 		stream
	// 	);

	set_view_dir(vec3(-0.022,-0.669,-0.743));
	set_scale(2.416f);
	tlog::success() << "Loaded scene";

}

/////////////////////////////////////////////////////
// nerf
/////////////////////////////////////////////////////

void Testbed::update_density_grid_mean_and_bitfield_geometry(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS();

	size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
	m_geometry.nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_geometry.nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_geometry.nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	reduce_sum(m_geometry.nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_geometry.nerf.density_grid_mean.data(), n_elements, stream);

	linear_kernel(grid_to_bitfield_geometry, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_geometry.nerf.max_cascade + 1), m_geometry.nerf.density_grid.data(), m_geometry.nerf.density_grid_bitfield.data(), m_geometry.nerf.density_grid_mean.data());

	for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
		linear_kernel(bitfield_max_pool_geometry, 0, stream, n_elements/64, m_geometry.nerf.get_density_grid_bitfield_mip(level-1), m_geometry.nerf.get_density_grid_bitfield_mip(level));
	}

	set_all_devices_dirty();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// shading functions
///////////////////////////////////////////////////////////////////////////////////////////////////

// void Testbed::generate_origin_samples(vec3* positions, uint32_t n_to_generate, cudaStream_t stream) {

// 	generate_random_uniform<float>(stream, m_rng, n_to_generate*3, (float*)positions);

// 	linear_kernel(sample_uniform_on_triangle_kernel_geometry, 0, stream,
// 		n_to_generate,
// 		m_geometry.mesh_cpu[0].sdf.triangle_cdf.data(),
// 		(uint32_t)m_geometry.mesh_cpu[0].sdf.triangle_cdf.size(),
// 		m_geometry.mesh_cpu[0].sdf.triangles_gpu.data(),
// 		positions
// 	);

// 	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
// }

// void Testbed::generate_direction_samples(vec3* direction, uint32_t n_to_generate, cudaStream_t stream) {

// 	GPUMemory<float> u (n_to_generate*2);
// 	generate_random_uniform<float>(stream, m_rng, n_to_generate*2, (float*)u);

// 	direction = cosine_hemisphere(vec2(u[0], u[1]));

// 	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
// }

}