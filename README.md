# tinybvh
Single-header BVH construction and traversal library written as "Sane C++" (or "C with classes"). The library has no dependencies. 

# BVH?
A Bounding Volume Hierarchy is a data structure used to quickly find intersections in a virtual scene; most commonly between a ray and a group of triangles. You can read more about this in a series of articles on the subject: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics .

Right now tiny_bvh comes with two builders:
* ````BVH::Build```` : Efficient plain-C/C+ binned SAH BVH builder which should run on any platform.
* ````BVH::BuildAVX```` : A highly optimized version of BVH::Build for Intel CPUs.

Once a BVH is constructed, it may be _refitted_ in case the triangles moved using ````BVH::Refit````. Refitting is substantially faster than rebuilding and works well if the animation is subtle. Refitting does not work if polygon counts change.

A constructed BVH can be used to quickly intersect a ray with the geometry, using ````BVH::Intersect````.

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

# Version 0.5.1
This version of the library includes the following functionality:
* Binned SAH BVH builder
* Fast binned SAH BVH builder using AVX intrinsics
* Spatial Splits (SBVH) builder
* BVH optimizer: reduces SAH cost and improves ray tracing performance
* Collapse to 4-wide and 8-wide BVH
* Conversion of 4-wide BVH to GPU-friendly 64-byte quantized format
* Single-ray and packet traversal.

The current version of the library is rapidly gaining functionality. Please expect changes to the interface.

Plans:

* OpenCL traversal example
* 'Compressed Wide BVH' data structure (CWBVH)
* Efficient CWBVH GPU traversal
* TLAS/BLAS traversal with BLAS transforms
  
These features have already been completed but need polishing and adapting to the interface, once it is settled. CWBVH GPU traversal combined with an optimized SBVH provides state-of-the-art **#RTXOff** performance; expect _billions of rays per second_.

# Contact
Questions, remarks? Contact me at bikker.j@gmail.com or on twitter: @j_bikker, or BlueSky: @jbikker.bsky.social .

# License
This library is made available under the MIT license, which starts as follows: "Permission is hereby granted, free of charge, .. , to deal in the Software **without restriction**". Enjoy.
