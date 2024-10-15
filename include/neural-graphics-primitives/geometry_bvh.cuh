/** @file   geometry_bvh.cuh
 *  @author Fatemeh Salehi
 */

#pragma once

#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/mesh.h>

#include <memory>

namespace ngp {

enum class NodeType : int {
        MESH,
        NERF
};

enum class BvhType : int {
        MESH,
        NERF
};

struct GeometryBvhNode {

	BoundingBox bb;
	int left_idx; // negative values indicate leaves
	int right_idx;
	NodeType type;	
};

using FixedIntStack = FixedStack<int>;


__host__ __device__ std::pair<int, float> geometrybvh_ray_intersect(const vec3& ro, const vec3& rd, const GeometryBvhNode* __restrict__ bvhnodes);

class GeometryBvh {
public:
	virtual void signed_distance_gpu_mesh(uint32_t n_elements, EMeshSdfMode mode, const vec3* gpu_positions, float* gpu_distances, const MeshData* __restrict__ meshes, bool use_existing_distances_as_upper_bounds, cudaStream_t stream) = 0;
	virtual void ray_trace_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const MeshData* __restrict__ meshes, const Nerf* __restrict__ nerfs, cudaStream_t stream) = 0;
	virtual void ray_trace_mesh_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const MeshData* __restrict__ meshes, cudaStream_t stream) = 0;
	virtual void build_mesh(std::vector<MeshData>& meshes, uint32_t n_primitives_per_leaf) = 0;
	virtual void build_nerf(std::vector<Nerf>& nerfs, uint32_t n_primitives_per_leaf) = 0;
	virtual void build_optix(const GPUMemory<GeometryBvhNode>& nodes, cudaStream_t stream) = 0;

	static std::unique_ptr<GeometryBvh> make();

	GeometryBvhNode* nodes_gpu() const {
		return m_nodes_gpu.data();
	}
	
	GeometryBvhNode* get_nodes() {
		return m_nodes.data();
	}

	BvhType nodes_type() const {
		return m_nodes_type;
	}

	void set_nodes_type(BvhType type) {
		m_nodes_type = type;
	}

	const std::vector<const TriangleBvhNode*>& getBvhNodes() const {
        return m_bvhnodes;
    }

    void setBvhNodes(const std::vector<const TriangleBvhNode*>& nodes) {
        m_bvhnodes = nodes;
    }

    const std::vector<const Triangle*>& getTriangles() const {
        return m_triangles;
    }

    void setTriangles(const std::vector<const Triangle*>& tris) {
        m_triangles = tris;
    }
	GPUMemory<const TriangleBvhNode*> m_bvhnodes_gpu;
	GPUMemory<const Triangle*> m_triangles_gpu;
	std::vector<const TriangleBvhNode*> m_bvhnodes;
	std::vector<const Triangle*> m_triangles;
	
protected:
	// if I want to keep the bvh seperate, I can add the type to the bvh.
	std::vector<GeometryBvhNode> m_nodes;
	GPUMemory<GeometryBvhNode> m_nodes_gpu;
	
	GeometryBvh(){}

	BvhType m_nodes_type;
};


}
