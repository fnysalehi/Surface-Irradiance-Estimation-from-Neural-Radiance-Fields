/** @file   geometry_bvh.cu
 *  @author Fatemeh Salehi
 */

#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/geometry_bvh.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

#include <stack>

namespace ngp {

constexpr float MAX_DIST = 100.0f;

__device__ __host__ int hit_counter = 0;

NGP_HOST_DEVICE BoundingBox::BoundingBox(MeshData* begin, MeshData* end) {
    // Initialize the bounding box to the first point of the first triangle of the first mesh
    min = max = begin->sdf.triangles_cpu[0].a;
    for (auto it = begin; it != end; ++it) {
        enlarge(*it);
    }
}

NGP_HOST_DEVICE void BoundingBox::enlarge(const MeshData& mesh) {
	// add the translation here insted of in the build, for now I assume the center for meshes are not considered!
	for (const Triangle& triangle : mesh.sdf.triangles_cpu) {
	    // Enlarge the bounding box to include the current triangle's points
	    enlarge(triangle.a);
	    enlarge(triangle.b);
	    enlarge(triangle.c);

	}
}

NGP_HOST_DEVICE BoundingBox::BoundingBox(Nerf* begin, Nerf* end) {
	min = max = begin->center;
	inflate(begin->scale);
	for (auto it = begin; it != end; ++it) {
		enlarge(*it);
	}
}

NGP_HOST_DEVICE void BoundingBox::enlarge(const Nerf& other){
	BoundingBox otherBox(other.center, other.center);
    otherBox.inflate(other.scale);
	enlarge(otherBox);
}

__global__ void signed_distance_watertight_kernel_geometry(uint32_t n_elements, const vec3* __restrict__ positions, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float* __restrict__ distances, bool use_existing_distances_as_upper_bounds = false);
__global__ void signed_distance_raystab_kernel_geometry(uint32_t n_elements, const vec3* __restrict__ positions, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float* __restrict__ distances, bool use_existing_distances_as_upper_bounds = false);
__global__ void unsigned_distance_kernel_geometry(uint32_t n_elements, const vec3* __restrict__ positions, const MeshData* __restrict__ meshes, float* __restrict__ distances, bool use_existing_distances_as_upper_bounds = false);

__global__ void mesh_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const TriangleBvhNode** __restrict__ bvhnodes, const Triangle** __restrict__ triangles, const MeshData* __restrict__ meshes);		
// __global__ void nerf_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const Nerf* __restrict__ nerfs);
template <uint32_t BRANCHING_FACTOR>
class GeometryBvhWithBranchingFactor : public GeometryBvh {
public:

	__host__ __device__ static std::pair<int, float> ray_intersect_triangle(const vec3& ro, const vec3& rd, const TriangleBvhNode* bvhnodes,  const Triangle* triangles) {
		
		FixedIntStack query_stack;
		query_stack.push(0);

		float mint = MAX_DIST;
		int shortest_idx = -1;

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const TriangleBvhNode& node = bvhnodes[idx];
			// checks if it's a leaf node
			if (node.left_idx < 0) {
				// checks each triangle in the leaf node for intersection with the ray, updating mint and shortest_idx if a closer intersection is found
				int end = -node.right_idx-1;
				for (int i = -node.left_idx-1; i < end; ++i) {
					float t = triangles[i].ray_intersect(ro, rd);
					if (t < mint) {
						mint = t;
						shortest_idx = i;
					}
				}
			} 
			// calculates the intersection of the ray with the bounding boxes of the node's children and sorts them by distance
			else {
				DistAndIdx children[BRANCHING_FACTOR];

				uint32_t first_child = node.left_idx;

				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					children[i] = {bvhnodes[i+first_child].bb.ray_intersect(ro, rd).x, i+first_child};
				}

				sorting_network<BRANCHING_FACTOR>(children);

				// pushes the indices ofchildren with the closest bounding boxes (intersect with the ray) to the query stack
				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					if (children[i].dist < mint) {
						query_stack.push(children[i].idx);
					}
				}
			}
		}

		return {shortest_idx, mint};
	}
	__host__ __device__ static std::pair<int, float> ray_intersect_triangle(const vec3& ro, const vec3& rd, const MeshData* __restrict__ mesh) {
		const Triangle* triangles = mesh->sdf.triangles_gpu.data();
		const TriangleBvhNode* bvhnodes = mesh->sdf.triangle_bvh->nodes_gpu();
		
		printf("Hello from the ray_intersect_triangle mesh kernel!\n");
		
		FixedIntStack query_stack;
		query_stack.push(0);

		float mint = MAX_DIST;
		int shortest_idx = -1;
		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const TriangleBvhNode& node = bvhnodes[idx];
			printf("node.left_idx: %d\n", node.left_idx);
			// checks if it's a leaf node
			if (node.left_idx < 0) {
				printf("node.left_idx: %d\n", node.left_idx);
				// checks each triangle in the leaf node for intersection with the ray, updating mint and shortest_idx if a closer intersection is found
				int end = -node.right_idx-1;
				for (int i = -node.left_idx-1; i < end; ++i) {
					float t = triangles[i].ray_intersect(ro, rd);
					if (t < mint) {
						mint = t;
						shortest_idx = i;
					}
				}
			} 
			// calculates the intersection of the ray with the bounding boxes of the node's children and sorts them by distance
			else {
				DistAndIdx children[BRANCHING_FACTOR];

				uint32_t first_child = node.left_idx;

				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					children[i] = {bvhnodes[i+first_child].bb.ray_intersect(ro, rd).x, i+first_child};
				}

				sorting_network<BRANCHING_FACTOR>(children);

				// pushes the indices ofchildren with the closest bounding boxes (intersect with the ray) to the query stack
				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					if (children[i].dist < mint) {
						query_stack.push(children[i].idx);
					}
				}
			}
		}

		return {shortest_idx, mint};
	}


	__host__ __device__ static std::tuple<int, float, vec3> ray_intersect(const vec3& ro, const vec3& rd, const GeometryBvhNode* __restrict__ meshbvhnodes) {
		
		float mint = MAX_DIST;
		float maxt = MAX_DIST;
		int mesh_idx = -1;
		int index = -1;
		auto root = meshbvhnodes[0];
		int start =  root.left_idx -1;
		int end =  root.right_idx;
		vec3 normal = vec3(0.0f);

		for (int idx = start; idx < end; ++idx) {
			const GeometryBvhNode& node = meshbvhnodes[idx];
			if(node.left_idx < 0 ) {
				if(node.left_idx != node.right_idx) {

					// float t =  node.bb.ray_intersect(ro, rd).x;
					vec2 t = node.bb.ray_intersect(ro, rd);
					if (t.x < mint && t.x > - std::numeric_limits<float>::max()) {
						index = idx;
						mint = t.x;
						mesh_idx = -node.left_idx-1;
						if(t.y < std::numeric_limits<float>::max())
							maxt = t.y;
						else
							maxt = 1.0f;
					}
				}
			}
		}

		auto p = ro + maxt * rd;
		normal = meshbvhnodes[index].bb.normal(p);
		return {mesh_idx, maxt, normal};
	}

	__host__ __device__ static std::pair<int, float> ray_intersect(const vec3& ro, const vec3& rd, const GeometryBvhNode* __restrict__ nerfbvhnodes, const Nerf* __restrict__ nerfs) {
		FixedIntStack query_stack;
		query_stack.push(0);

		float mint = MAX_DIST;
		int shortest_idx = -1;
		
		printf("Hello from the nerf kernel!\n");
		
		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const GeometryBvhNode& node = nerfbvhnodes[idx];
			
			float t = node.bb.ray_intersect(ro, rd).x;

        	// If the ray intersects the bounding box of the node
			if(t < std::numeric_limits<float>::max())
			{	

				// If it's a leaf node
				if (node.left_idx < 0) {
					if (t < mint) {
						mint = t;
						shortest_idx = -node.left_idx-1;	//not sure
					}		
				}
            	// If it's not a leaf node
				// same as trinagle bvh
				else {
					DistAndIdx children[BRANCHING_FACTOR];

					uint32_t first_child = node.left_idx;

					NGP_PRAGMA_UNROLL
					for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
						children[i] = {nerfbvhnodes[i+first_child].bb.ray_intersect(ro, rd).x, i+first_child};
					}

					sorting_network<BRANCHING_FACTOR>(children);

					// pushes the indices ofchildren with the closest bounding boxes (intersect with the ray) to the query stack
					NGP_PRAGMA_UNROLL
					for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
						if (children[i].dist < mint) {
							query_stack.push(children[i].idx);
						}
					}
				}
			}
		}

		return {shortest_idx, mint};
	}

	__host__ __device__ static std::pair<int, float> closest_triangle(const vec3& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq) {
		FixedIntStack query_stack;
		query_stack.push(0);

		float shortest_distance_sq = max_distance_sq;
		int shortest_idx = -1;

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const TriangleBvhNode& node = bvhnodes[idx];

			if (node.left_idx < 0) {
				int end = -node.right_idx-1;
				for (int i = -node.left_idx-1; i < end; ++i) {
					float dist_sq = triangles[i].distance_sq(point);
					if (dist_sq <= shortest_distance_sq) {
						shortest_distance_sq = dist_sq;
						shortest_idx = i;
					}
				}
			} else {
				DistAndIdx children[BRANCHING_FACTOR];

				uint32_t first_child = node.left_idx;

				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					children[i] = {bvhnodes[i+first_child].bb.distance_sq(point), i+first_child};
				}

				sorting_network<BRANCHING_FACTOR>(children);

				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					if (children[i].dist <= shortest_distance_sq) {
						query_stack.push(children[i].idx);
					}
				}
			}
		}

		if (shortest_idx == -1) {
			// printf("No closest triangle found. This must be a bug! %d\n", BRANCHING_FACTOR);
			shortest_idx = 0;
			shortest_distance_sq = 0.0f;
		}

		return {shortest_idx, std::sqrt(shortest_distance_sq)};
	}

	// Assumes that "point" is a location on a triangle
	__host__ __device__ static vec3 avg_normal_around_point(const vec3& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
		FixedIntStack query_stack;
		query_stack.push(0);

		static constexpr float EPSILON = 1e-6f;

		float total_weight = 0;
		vec3 result = vec3(0.0f);

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const TriangleBvhNode& node = bvhnodes[idx];

			if (node.left_idx < 0) {
				int end = -node.right_idx-1;
				for (int i = -node.left_idx-1; i < end; ++i) {
					if (triangles[i].distance_sq(point) < EPSILON) {
						float weight = 1; // TODO: cot weight
						result += triangles[i].normal();
						total_weight += weight;
					}
				}
			} else {
				uint32_t first_child = node.left_idx;

				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					if (bvhnodes[i+first_child].bb.distance_sq(point) < EPSILON) {
						query_stack.push(i+first_child);
					}
				}
			}
		}

		return result / total_weight;
	}

	__host__ __device__ static float unsigned_distance(const vec3& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq) {
		return closest_triangle(point, bvhnodes, triangles, max_distance_sq).second;
	}

	__host__ __device__ static float signed_distance_watertight(const vec3& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq) {
		auto p = closest_triangle(point, bvhnodes, triangles, max_distance_sq);

		const Triangle& tri = triangles[p.first];
		vec3 closest_point = tri.closest_point(point);
		vec3 avg_normal = avg_normal_around_point(closest_point, bvhnodes, triangles);

		return copysign(p.second, dot(avg_normal, point - closest_point));
	}

	__host__ __device__ static float signed_distance_raystab(const vec3& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq, default_rng_t rng={}) {
		float distance = unsigned_distance(point, bvhnodes, triangles, max_distance_sq);

		vec2 offset = random_val_2d(rng);

		static constexpr uint32_t N_STAB_RAYS = 32;
		for (uint32_t i = 0; i < N_STAB_RAYS; ++i) {
			// Use a Fibonacci lattice (with random offset) to regularly
			// distribute the stab rays over the sphere.
			vec3 d = fibonacci_dir<N_STAB_RAYS>(i, offset);

			// If any of the stab rays goes outside the mesh, the SDF is positive.
			// if (ray_intersect_triangle(point, d, bvhnodes, triangles).first < 0) {
			// 	return distance;
			// }
		}

		return -distance;
	}

	// Assumes that "point" is a location on a triangle
	vec3 avg_normal_around_point(const vec3& point, const Triangle* __restrict__ triangles) const {
		return avg_normal_around_point(point, m_nodes.data(), triangles);
	}

	void signed_distance_gpu_mesh(uint32_t n_elements, EMeshSdfMode mode, const vec3* gpu_positions, float* gpu_distances, const MeshData* __restrict__ meshes, bool use_existing_distances_as_upper_bounds, cudaStream_t stream) override {
		
		const auto mesh = meshes[0];
		const TriangleBvhNode* bvhnodes = mesh.sdf.triangle_bvh->nodes_gpu();
		const Triangle* triangles = mesh.sdf.triangles_gpu.data();

		if (mode == EMeshSdfMode::Watertight) {
			linear_kernel(signed_distance_watertight_kernel_geometry, 0, stream,
				n_elements,
				gpu_positions,
				bvhnodes,
				triangles,
				gpu_distances,
				use_existing_distances_as_upper_bounds
			);
		} else {
			if (mode == EMeshSdfMode::Raystab) {
				linear_kernel(signed_distance_raystab_kernel_geometry, 0, stream,
					n_elements,
					gpu_positions,
					bvhnodes,
					triangles,
					gpu_distances,
					use_existing_distances_as_upper_bounds
				);
			} else if (mode == EMeshSdfMode::PathEscape) {
				throw std::runtime_error{"TriangleBvh: EMeshSdfMode::PathEscape is only supported with OptiX enabled."};
			}
		}
	}
	
	
	void ray_trace_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const MeshData* __restrict__ meshes, const Nerf* __restrict__ nerfs, cudaStream_t stream) override {
		linear_kernel(mesh_raytrace_kernel, 0, stream,
			n_elements,
			gpu_positions,
			gpu_directions,
			m_nodes_gpu.data(),
			m_bvhnodes_gpu.data(),
			m_triangles_gpu.data(),
			meshes	
		);

		int hit_counter_host = 0;
		cudaMemcpyFromSymbol(&hit_counter_host, hit_counter, sizeof(int));
		printf("Number of rays that hit a mesh: %d\n", hit_counter_host);
	}

	void ray_trace_mesh_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const MeshData* __restrict__ meshes, cudaStream_t stream) override {
		linear_kernel(mesh_raytrace_kernel, 0, stream,
			n_elements,
			gpu_positions,
			gpu_directions,
			m_nodes_gpu.data(),
			m_bvhnodes_gpu.data(),
			m_triangles_gpu.data(),
			meshes	
		);	
	}

	void build_mesh(std::vector<MeshData>& meshes, uint32_t n_primitives_per_leaf) override {
		m_nodes.clear();

		tlog::info() << "Building Mesh GeometryBvh with branching factor " << BRANCHING_FACTOR;
		
		// Root
		m_nodes.emplace_back();
		auto bb = BoundingBox(meshes.data(), meshes.data() + meshes.size());
		m_nodes.front().bb = bb;
		tlog::info() << " main aabb=" <<bb;


		struct BuildNode {
			int node_idx;
			std::vector<MeshData>::iterator begin;
			std::vector<MeshData>::iterator end;
		};

		std::stack<BuildNode> build_stack;
		build_stack.push({0, std::begin(meshes), std::end(meshes)});

		while (!build_stack.empty()) {
			const BuildNode& curr = build_stack.top();
			size_t node_idx = curr.node_idx;

			std::array<BuildNode, BRANCHING_FACTOR> c;
			c[0].begin = curr.begin;
			c[0].end = curr.end;

			build_stack.pop();

			// Partition the triangles into the children
			int number_c = 1;
			while (number_c < BRANCHING_FACTOR) {
				for (int i = number_c - 1; i >= 0; --i) {
					auto& child = c[i];

					// Choose axis with maximum standard deviation
					vec3 mean = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						mean += it->center; // In the traingle bvh they use centroid instead of center!
					}
					mean /= (float)std::distance(child.begin, child.end);

					vec3 var = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						vec3 diff = it->center - mean;
						var += diff * diff;
					}
					var /= (float)std::distance(child.begin, child.end);

					float max_val = max(var);
					int axis = var.x == max_val ? 0 : (var.y == max_val ? 1 : 2);

					auto m = child.begin + std::distance(child.begin, child.end)/2;
					std::nth_element(child.begin, m, child.end, [&](const MeshData& mesh1, const MeshData& mesh2) { return mesh1.center[0]+mesh1.center[1]+mesh1.center[2] < mesh2.center[0]+mesh2.center[1]+mesh2.center[2]; });

					c[i*2].begin = c[i].begin;
					c[i*2+1].end = c[i].end;
					c[i*2].end = c[i*2+1].begin = m;
				}

				number_c *= 2;
			}

			// Create next build nodes
			m_nodes[node_idx].left_idx = (int)m_nodes.size();
			for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
				auto& child = c[i];
				assert(child.begin != child.end);
				child.node_idx = (int)m_nodes.size();

				m_nodes.emplace_back();
				m_nodes.back().bb = BoundingBox(&*child.begin, &*child.end);

				if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {

					m_nodes.back().left_idx = -(int)std::distance(std::begin(meshes), child.begin)-1;
					m_nodes.back().right_idx = -(int)std::distance(std::begin(meshes), child.end)-1;
				} else {
					build_stack.push(child);
				}
			}
			m_nodes[node_idx].right_idx = (int)m_nodes.size();
		}

		m_nodes_gpu.resize_and_copy_from_host(m_nodes);

		tlog::success() << "Built GeometryBvh: nodes=" << m_nodes.size();
	}

	void build_nerf(std::vector<Nerf>& nerfs, uint32_t n_primitives_per_leaf) override {
		m_nodes.clear();

		tlog::info() << "Building Nerf GeometryBvh with branching factor " << BRANCHING_FACTOR;
		
		// Root
		m_nodes.emplace_back();
		auto bb = BoundingBox(nerfs.data(), nerfs.data() + nerfs.size());
		m_nodes.front().bb = bb;
		tlog::info() << " main aabb=" <<bb;


		struct BuildNode {
			int node_idx;
			std::vector<Nerf>::iterator begin;
			std::vector<Nerf>::iterator end;
		};

		std::stack<BuildNode> build_stack;
		build_stack.push({0, std::begin(nerfs), std::end(nerfs)});

		while (!build_stack.empty()) {
			const BuildNode& curr = build_stack.top();
			size_t node_idx = curr.node_idx;

			std::array<BuildNode, BRANCHING_FACTOR> c;
			c[0].begin = curr.begin;
			c[0].end = curr.end;

			build_stack.pop();

			// Partition the triangles into the children
			int number_c = 1;
			while (number_c < BRANCHING_FACTOR) {
				for (int i = number_c - 1; i >= 0; --i) {
					auto& child = c[i];

					// Choose axis with maximum standard deviation
					vec3 mean = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						mean += it->center; // In the traingle bvh they use centroid instead of center!
					}
					mean /= (float)std::distance(child.begin, child.end);

					vec3 var = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						vec3 diff = it->center - mean;
						var += diff * diff;
					}
					var /= (float)std::distance(child.begin, child.end);

					float max_val = max(var);
					int axis = var.x == max_val ? 0 : (var.y == max_val ? 1 : 2);

					auto m = child.begin + std::distance(child.begin, child.end)/2;
					std::nth_element(child.begin, m, child.end, [&](const Nerf& nerf1, const Nerf& nerf2) { return nerf1.center[0]+nerf1.center[1]+nerf1.center[2] < nerf2.center[0]+nerf2.center[1]+nerf2.center[2]; });

					c[i*2].begin = c[i].begin;
					c[i*2+1].end = c[i].end;
					c[i*2].end = c[i*2+1].begin = m;
				}

				number_c *= 2;
			}

			// Create next build nodes
			m_nodes[node_idx].left_idx = (int)m_nodes.size();
			for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
				auto& child = c[i];
				assert(child.begin != child.end);
				child.node_idx = (int)m_nodes.size();

				m_nodes.emplace_back();
				m_nodes.back().bb = BoundingBox(&*child.begin, &*child.end);
				tlog::info() << " aabb=" << m_nodes.back().bb;


				if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
					m_nodes.back().left_idx = -(int)std::distance(std::begin(nerfs), child.begin)-1;
					m_nodes.back().right_idx = -(int)std::distance(std::begin(nerfs), child.end)-1;
				} else {
					build_stack.push(child);
				}
			}
			m_nodes[node_idx].right_idx = (int)m_nodes.size();
		}

		m_nodes_gpu.resize_and_copy_from_host(m_nodes);

		tlog::success() << "Built GeometryBvh: nodes=" << m_nodes.size();
	}


	void build_optix(const GPUMemory<GeometryBvhNode>& nodes, cudaStream_t stream) override {

	}

	GeometryBvhWithBranchingFactor() {}

};

using GeometryBvh4 = GeometryBvhWithBranchingFactor<4>;
using GeometryBvh1 = GeometryBvhWithBranchingFactor<2>;

std::unique_ptr<GeometryBvh> GeometryBvh::make() {
	return std::unique_ptr<GeometryBvh>(new GeometryBvh4());
}

// each thread processes one ray.
__global__ void mesh_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const TriangleBvhNode** __restrict__ bvhnodes, const Triangle** __restrict__ triangles, const MeshData* __restrict__ meshes) {		
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	// no more rays to process
	if (i >= n_elements) return;

	auto pos = positions[i];
	auto dir = directions[i];
	
	// first element is the index of the intersected mesh
	// second element is the distance to the intersection to the bounding box of the mesh
	auto p = GeometryBvh4::ray_intersect(pos, dir, nodes);

	if (std::get<0>(p) > -1) {
	
		const TriangleBvhNode* mesh_bvhnodes = bvhnodes[std::get<0>(p)];
        const Triangle* mesh_triangles = triangles[std::get<0>(p)];
		
		auto result = GeometryBvh4::ray_intersect_triangle(pos, dir, mesh_bvhnodes, mesh_triangles);
		
		
		positions[i] = pos + result.second * dir;
		if(result.first > -1) {
			directions[i] = mesh_triangles[result.first].normal();
		}

		// else {
		// 	// positions[i] = pos + std::get<1>(p) * dir;
		// 	directions[i] = dir;
		// }
	}
}

__global__ void signed_distance_watertight_kernel_geometry(uint32_t n_elements,
	const vec3* __restrict__ positions,
	const TriangleBvhNode* __restrict__ bvhnodes,
	const Triangle* __restrict__ triangles,
	float* __restrict__ distances,
	bool use_existing_distances_as_upper_bounds
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
	distances[i] = GeometryBvh4::signed_distance_watertight(positions[i], bvhnodes, triangles, max_distance*max_distance);
}

__global__ void signed_distance_raystab_kernel_geometry(
	uint32_t n_elements,
	const vec3* __restrict__ positions,
	const TriangleBvhNode* __restrict__ bvhnodes,
	const Triangle* __restrict__ triangles,
	float* __restrict__ distances,
	bool use_existing_distances_as_upper_bounds
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	
	float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
	default_rng_t rng;
	rng.advance(i * 2);

	distances[i] = GeometryBvh4::signed_distance_raystab(positions[i], bvhnodes, triangles, max_distance*max_distance, rng);
}

__global__ void unsigned_distance_kernel_geometry(uint32_t n_elements,
	const vec3* __restrict__ positions,
	const TriangleBvhNode* __restrict__ bvhnodes,
	const Triangle* __restrict__ triangles,
	float* __restrict__ distances,
	bool use_existing_distances_as_upper_bounds
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
	distances[i] = GeometryBvh4::unsigned_distance(positions[i], bvhnodes, triangles, max_distance*max_distance);
}

}


