# tinybvh
Single-header BVH construction and traversal library written as "Sane C++" (or "C with classes"). The library has no dependencies. 

# tinyocl
Single-header OpenCL library, which helps you select and initialize a device. It also loads, compiles and runs kernels, with several convenient features:
* Include-file expansion for AMD devices
* Multi-argument passing
* Host/device buffer management
* Vendor and architecture detection and propagation to #defines in OpenCL code
* ..And many other things.

To use tinyocl, just include ````tiny_ocl.h````; this will automatically cause linking with ````OpenCL.lib```` in the 'external' folder, which in turn passes on work to vendor-specific driver code. But all that is not your problem!

Note that the ````tiny_bvh.h```` library will work without ````tiny_ocl.h```` and remains dependency-free. The new ````tiny_ocl.h```` is only needed in projects that wish to trace rays _on the GPU_ using BVHs created by ````tiny_bvh.h````. 
  
# BVH?
A Bounding Volume Hierarchy is a data structure used to quickly find intersections in a virtual scene; most commonly between a ray and a group of triangles. You can read more about this in a series of articles on the subject: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics .

Right now tiny_bvh comes with four builders:
* ````BVH::Build```` : Efficient plain-C/C+ binned SAH BVH builder which should run on any platform.
* ````BVH::BuildAVX```` : A highly optimized version of BVH::Build for Intel CPUs.
* ````BVH::BuildNEON```` : An optimized version of BVH::Build for ARM/NEON.
* ````BVH::BuildHQ```` : A 'spatial splits' BVH builder, for highest BVH quality.

A constructed BVH can be used to quickly intersect a ray with the geometry, using ````BVH::Intersect````.

The constructed BVH will have a layout suitable for construction ('````WALD_32BYTE````'). Several other layouts for the same data are available, which all serve one or more specific purposes. You can convert between layouts using ````BVH::Convert````. The available layouts are:
* ````BVH::WALD_32BYTE```` : A compact format that stores the AABB for a node, along with child pointers and leaf information in a cross-platform-friendly way. The 32-byte size allows for cache-line alignment.
* ````BVH::VERBOSE```` : A format designed for modifying BVHs, e.g. for post-build optimizations using ````BVH::Optimize()````.
* ````BVH::AILA_LAINE```` : This format uses 64 bytes per node and stores the AABBs of the two child nodes. This is the format recommended for basic GPU ray tracing.
* ````BVH::BASIC_BVH4```` : In this format, each node stores four child pointers, reducing the depth of the tree. This improves performance for divergent rays.
* ````BVH::BVH4_GPU```` : The ````BASIC_BVH4```` format can be converted to the more compact ````BVH4_GPU```` layout, which will be faster for GPU ray tracing.
* ````BVH::BASIC_BVH8```` : This format stores eight child pointers, further reducing the depth of the tree. The only purpose is the construction of ````BVH::CWBVH````.
* ````BVH::CWBVH```` : An advanced 80-byte representation of the 8-wide BVH, for state-of-the-art GPU rendering, based on the [2017 paper](https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs) by Ylitie et al.

A BVH in the ````BVH::WALD_32BYTE```` format may be _refitted_ in case the triangles moved using ````BVH::Refit````. Refitting is substantially faster than rebuilding and works well if the animation is subtle. Refitting does not work if polygon counts change.

# How To Use
The library ````tiny_bvh.h```` is designed to be easy to use. Please have a look at tiny_bvh_test.cpp for an example. A Visual Studio 'solution' (.sln/.vcxproj) is included, as well as a CMake file. That being said: The examples consists of only a single source file, which can be compiled with clang or g++, e.g.:

````g++ -std=c++20 -mavx tiny_bvh_test.cpp -o tiny_bvh_test````

The single-source sample **ASCII test renderer** can be compiled with

````g++ -std=c++20 -mavx tiny_bvh_renderer.cpp -o tiny_bvh_renderer````

The cross-platform fenster-based single-source **bitmap renderer** can be compiled with

````g++ -std=c++20 -mavx -mwindows -O3 tiny_bvh_fenster.cpp -o tiny_bvh_fenster```` (on windows)

```g++ -std=c++20 -mavx -O3 -framework Cocoa tiny_bvh_fenster.cpp -o tiny_bvh_fenster``` (on macOS)

The **performance measurement tool** use OpenMP and can be compiled with:

````g++ -std=c++20 -mavx -Ofast -fopenmp tiny_bvh_speedtest.cpp -o tiny_bvh_speedtest````

# Version 0.9.5
This version of the library includes the following functionality:
* Binned SAH BVH builder
* Fast binned SAH BVH builder using AVX intrinsics
* Fast binned SAH BVH builder using NEON intrinsices, by [wuyakuma](https://github.com/wuyakuma)
* Spatial Splits (SBVH) builder
* 'Compressed Wide BVH' (CWBVH) data structure
* BVH optimizer: reduces SAH cost and improves ray tracing performance
* Collapse to 4-wide and 8-wide BVH
* Conversion of 4-wide BVH to GPU-friendly 64-byte quantized format
* Single-ray and packet traversal
* OpenCL traversal: Aila & Laine, 4-way quantized, CWBVH
* Support for WASM / EMSCRIPTEN, g++, clang, Visual Studio.
* Optional user-defined memory allocation, by [Thierry Cantenot](https://github.com/tcantenot). 

The current version of the library is rapidly gaining functionality. Please expect changes to the interface.

Plans, ordered by priority:

* Interface:
  * Shadow ray queries
  * Large batch query (Optix-style)
* Documentation:
  * Wiki
  * Article on architecture and intended use
* Example renderers:
  * CPU WHitted-style ray tracer
  * GPU path tracer
  * GPU wavefront path tracer
* TLAS/BLAS traversal with BLAS transforms
  * Part of this: Build BVH over list of AABBs rather than tris
* BVH::Optimize:
  * Properly use C_trav and C_int for SAH **_(done)_**
  * Faster Optimize algorithm (complete paper implementation)
  * Understanding optimized SBVH performance
* CPU single-ray performance
  * Experiment with 4-wide layouts
  * Reverse-engineer Embree & PhysX
  
These features have already been completed but need polishing and adapting to the interface, once it is settled. CWBVH GPU traversal combined with an optimized SBVH provides state-of-the-art **#RTXOff** performance; expect _billions of rays per second_.

# Contact
Questions, remarks? Contact me at bikker.j@gmail.com or on twitter: @j_bikker, or BlueSky: @jbikker.bsky.social .

# License
This library is made available under the MIT license, which starts as follows: "Permission is hereby granted, free of charge, .. , to deal in the Software **without restriction**". Enjoy.
