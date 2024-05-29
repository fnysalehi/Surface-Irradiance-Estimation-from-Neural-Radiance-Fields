/** @file   geometry_bvh.cuh
 *  @author Fatemeh Salehi
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/sdf.h>
#include <neural-graphics-primitives/nerf_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>


namespace ngp {

struct DiscreteDistribution;
struct MeshData {
	
    Sdf sdf;
	    
	vec3 center = vec3(0.0f);

	};

}