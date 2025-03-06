# Hybrid NeRF-Mesh Rendering Framework

This repository contains an implementation of a **hybrid rendering framework** that integrates **Neural Radiance Fields (NeRF)** with **traditional polygonal mesh models**, enabling seamless and realistic scene rendering with a combination of volumetric and geometric representations. The project also incorporates **advanced shading techniques**, ensuring visually consistent lighting and materials across different data representations.

## **Project Overview**
NeRF has revolutionized novel view synthesis by modeling radiance fields from images, but its application in hybrid rendering scenarios with explicit geometry remains a challenge. This project addresses that gap by:
- Developing **efficient ray intersection algorithms** for handling both volumetric NeRF data and traditional mesh-based geometry.
- Implementing **precomputed environment maps** to enhance global illumination and shading accuracy.
- Optimizing **shadow ray tracing and importance sampling** to improve light transport.
- Applying **advanced shading techniques** to unify the visual properties of NeRF and mesh-based surfaces.
- Leveraging **GPU acceleration (CUDA)** to achieve real-time performance in hybrid rendering scenarios.

## **Key Features**
- **Seamless NeRF-Mesh Integration**: Supports rendering scenes that contain both NeRF-based volumetric models and explicit 3D meshes.
- **Advanced Shading**: Incorporates physically-based lighting models to blend NeRF and polygonal surfaces naturally.
- **Optimized Light Transport**: Uses shadow ray tracing and environment map precomputation to improve realism.
- **High-Performance Rendering**: Leverages CUDA-based acceleration to enable efficient real-time rendering.

## **Technologies & Tools**
- **Programming Languages:** C++, Python
- **Rendering & GPU Acceleration:**  CUDA
- **Machine Learning & NeRF Frameworks:** PyTorch, Tiny-CUDA-NN
- **3D Modeling & Data Processing:** Blender
